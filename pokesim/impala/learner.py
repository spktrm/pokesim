import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import collections

from copy import deepcopy
from dataclasses import asdict
from typing import Any, Mapping

from pokesim.nn.modelv2 import Model
from pokesim.structs import Batch, ModelOutput, State

from pokesim.impala.utils import EntropySchedule
from pokesim.impala.config import ImpalaConfig
from pokesim.utils import _print_params, SGDTowardsModel


def get_example(T: int, B: int, device: str):
    mask = torch.zeros(T, B, 12, dtype=torch.bool, device=device)
    mask[..., 0] = True
    return (
        torch.zeros(T, B, dtype=torch.long, device=device),
        torch.zeros(T, B, 4, dtype=torch.long, device=device),
        torch.zeros(T, B, 3, 6, 11, dtype=torch.long, device=device),
        torch.zeros(T, B, 2, 15, dtype=torch.long, device=device),
        torch.zeros(T, B, 2, 20, 2, dtype=torch.long, device=device),
        torch.zeros(T, B, 2, 7, dtype=torch.long, device=device),
        torch.zeros(T, B, 5, 3, dtype=torch.long, device=device),
        mask,
        torch.ones(T, B, 4, 3, dtype=torch.long, device=device),
        torch.ones(T, B, dtype=torch.bool, device=device),
    )


VTraceFromLogitsReturns = collections.namedtuple(
    "VTraceFromLogitsReturns",
    [
        "vs",
        "pg_advantages",
        "log_rhos",
        "behavior_action_log_probs",
        "target_action_log_probs",
    ],
)

VTraceReturns = collections.namedtuple("VTraceReturns", "vs pg_advantages")


def action_log_probs(policy_logits, actions):
    return torch.log(
        torch.gather(
            torch.flatten(policy_logits, 0, -2), -1, actions.view(-1, 1)
        ).view_as(actions)
    )


def from_logits(
    behavior_policy_logits,
    target_policy_logits,
    actions,
    discounts,
    rewards,
    values,
    bootstrap_value,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
):
    """V-trace for softmax policies."""

    target_action_log_probs = action_log_probs(target_policy_logits, actions)
    behavior_action_log_probs = action_log_probs(behavior_policy_logits, actions)
    log_rhos = target_action_log_probs - behavior_action_log_probs
    vtrace_returns = from_importance_weights(
        log_rhos=log_rhos,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold,
    )
    return VTraceFromLogitsReturns(
        log_rhos=log_rhos,
        behavior_action_log_probs=behavior_action_log_probs,
        target_action_log_probs=target_action_log_probs,
        **vtrace_returns._asdict(),
    )


@torch.no_grad()
def from_importance_weights(
    log_rhos,
    discounts,
    rewards,
    values,
    bootstrap_value,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
):
    """V-trace from log importance weights."""
    with torch.no_grad():
        rhos = torch.exp(log_rhos)
        if clip_rho_threshold is not None:
            clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
        else:
            clipped_rhos = rhos

        cs = torch.clamp(rhos, max=1.0)
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = torch.cat(
            [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
        )
        deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

        acc = torch.zeros_like(bootstrap_value)
        result = []
        for t in range(discounts.shape[0] - 1, -1, -1):
            acc = deltas[t] + discounts[t] * cs[t] * acc
            result.append(acc)
        result.reverse()
        vs_minus_v_xs = torch.stack(result)

        # Add V(x_s) to get v_s.
        vs = torch.add(vs_minus_v_xs, values)

        # Advantage for policy gradient.
        broadcasted_bootstrap_values = torch.ones_like(vs[0]) * bootstrap_value
        vs_t_plus_1 = torch.cat(
            [vs[1:], broadcasted_bootstrap_values.unsqueeze(0)], dim=0
        )
        if clip_pg_rho_threshold is not None:
            clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
        else:
            clipped_pg_rhos = rhos
        pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)

        # Make sure no gradients backpropagated through the returned values.
        return VTraceReturns(vs=vs, pg_advantages=pg_advantages)


def compute_baseline_loss(advantages, discounts):
    return 0.5 * torch.sum((advantages**2) * discounts)


def compute_policy_gradient_loss(policy, actions, advantages, discounts):
    cross_entropy = F.cross_entropy(
        torch.flatten(policy, 0, 1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach() * discounts)


class Learner:
    def __init__(
        self,
        config: ImpalaConfig = ImpalaConfig(),
        init: Mapping[str, Any] = None,
        use_amp: bool = False,
        trace_nets: bool = True,
        debug: bool = False,
    ):
        self.config = config
        self.use_amp = use_amp

        # Create initial parameters.
        self.params = Model()
        if init is not None:
            self.params.load_state_dict(init)

        self.params.train()

        self.extra_config = {
            "num_params": _print_params(self.params)[0],
            "entity_size": self.params.entity_size,
            "vector_size": self.params.vector_size,
        }

        self.params_actor = deepcopy(self.params).share_memory()
        self.params_actor.eval()

        self.params_actor_prev = deepcopy(self.params).share_memory()
        self.params_actor_prev.eval()

        self.params.to(self.config.learner_device)
        self.params_actor.to(self.config.actor_device)

        if not debug and trace_nets:
            actor_example = get_example(1, 1, self.config.actor_device)
            self.params_actor = torch.jit.trace(self.params_actor, actor_example)

            with torch.autocast(
                device_type=self.config.learner_device,
                dtype=torch.float16,
                enabled=self.use_amp,
            ):
                learner_example = get_example(1, 1, self.config.learner_device)

                self.params = torch.jit.trace(self.params, learner_example)

        # Parameter optimizers.
        self.optimizer = optim.Adam(
            self.params.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam.b1, self.config.adam.b2),
        )

        self.scaler = torch.cuda.amp.GradScaler()
        self.learner_steps = 0

    def save(self, fpath: str):
        torch.save(
            {
                "config": self.config,
                "params": self.params.state_dict(),
                "params_actor": self.params_actor.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "learner_steps": self.learner_steps,
            },
            fpath,
        )

    @classmethod
    def from_fpath(cls, fpath: str, ignore_config: bool = False, **kwargs):
        obj = cls(**kwargs)
        ckpt = torch.load(fpath, map_location="cpu")
        if not ignore_config:
            obj.config = ckpt["config"]
        obj.params.load_state_dict(ckpt["params"])
        obj.params_actor.load_state_dict(ckpt["params_actor"])
        obj.optimizer.load_state_dict(ckpt["optimizer"])
        obj.scaler.load_state_dict(ckpt["scaler"])
        obj.learner_steps = ckpt["learner_steps"]
        return obj

    def get_config(self):
        return {
            **asdict(self.config),
            **self.extra_config,
        }

    def _to_torch(self, arr: np.ndarray, device: str = None):
        if device is None:
            device = self.config.learner_device
        return torch.from_numpy(arr).to(device, non_blocking=True)

    def loss(self, batch: Batch) -> float:
        state = {
            **State(batch.state).dense(),
            "legal": batch.legal,
        }

        forward_batch = {
            key: self._to_torch(state[key])
            for key in [
                "turn",
                "active_moveset",
                "teams",
                "side_conditions",
                "volatile_status",
                "boosts",
                "field",
                "history",
                "legal",
            ]
        }

        learner_outputs: ModelOutput = self.params(**forward_batch)

        behavior_policy_logits = torch.from_numpy(batch.policy)
        target_policy_logits = learner_outputs.policy.cpu()
        action = torch.from_numpy(batch.action)
        discounts = torch.from_numpy(batch.valid).to(torch.float32)
        # rewards = torch.from_numpy(
        #     np.sign(batch.rewards) * ((batch.rewards / 100) ** 2)
        # ).squeeze(-1)
        rewards = torch.from_numpy(batch.rewards).squeeze(-1)
        values = learner_outputs.value.cpu().squeeze(-1)
        legal = self._to_torch(batch.legal)

        vtrace_returns = from_logits(
            behavior_policy_logits=behavior_policy_logits,
            target_policy_logits=target_policy_logits,
            actions=action,
            discounts=discounts,
            rewards=rewards,
            values=values * discounts,
            bootstrap_value=torch.zeros_like(values[-1]),
        )

        discounts = discounts.to(self.config.learner_device)

        pg_loss = compute_policy_gradient_loss(
            learner_outputs.policy,
            action.to(self.config.learner_device),
            vtrace_returns.pg_advantages.to(self.config.learner_device),
            discounts,
        )
        baseline_loss = compute_baseline_loss(
            vtrace_returns.vs.to(self.config.learner_device)
            - learner_outputs.value.squeeze(-1),
            discounts,
        )

        num_valid_actions = legal.sum(-1)
        entropy_loss = (
            (learner_outputs.policy * learner_outputs.log_policy).sum(-1)
            / torch.where(num_valid_actions == 1, 1, torch.log(num_valid_actions))
            * discounts
        ).sum()

        discounts_sum = discounts.sum().item()
        loss = pg_loss + 0.5 * baseline_loss + 1e-3 * entropy_loss
        loss = loss / discounts_sum

        self.scaler.scale(loss).backward()

        return {
            "v_loss": baseline_loss.item() / discounts_sum,
            "p_loss": pg_loss.item() / discounts_sum,
            "e_loss": entropy_loss.item() / discounts_sum,
        }

    def update_parameters(self, batch: Batch):
        """A jitted pure-functional part of the `step`."""

        loss_vals = self.loss(batch)

        self.scaler.unscale_(self.optimizer)

        norm = nn.utils.clip_grad.clip_grad_norm_(
            self.params.parameters(), self.config.clip_gradient
        )

        # Update `params`` using the computed gradient.
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.optimizer.zero_grad(set_to_none=True)

        self.params_actor.load_state_dict(self.params.state_dict())

        draws = abs(batch.rewards).sum(0).sum(1) == 0
        draw_ratio = draws.sum() / batch.valid.shape[1]

        logs = {
            **loss_vals,
            "draw_ratio": draw_ratio.item(),
            "norm": norm.item(),
        }
        return logs
