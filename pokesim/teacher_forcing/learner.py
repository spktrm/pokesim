import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import collections

from copy import deepcopy
from dataclasses import asdict
from typing import Any, Mapping
from pokesim.data import MODEL_INPUT_KEYS, NUM_HISTORY

from pokesim.nn.model import Model
from pokesim.structs import Batch, ModelOutput, State

from pokesim.teacher_forcing.config import ImpalaConfig
from pokesim.utils import _print_params, get_example


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
    policy = torch.flatten(policy_logits, 0, -2)
    return torch.log(torch.gather(policy, -1, actions.view(-1, 1)).view_as(actions))


def get_loss_entropy(
    policy: torch.Tensor, log_policy: torch.Tensor, legal: torch.Tensor
) -> torch.Tensor:
    loss_entropy = (policy * log_policy).sum(-1)
    num_legal_actions = legal.sum(-1)
    denom = torch.log(num_legal_actions)
    denom = torch.where(num_legal_actions <= 1, 1, denom)
    loss_entropy = loss_entropy / denom
    return loss_entropy


@torch.no_grad()
def from_logits(
    behavior_policy_logits,
    target_policy_logits,
    actions,
    rewards,
    discounts,
    values,
    bootstrap_value,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
):
    """V-trace for softmax policies."""

    target_action_log_probs = action_log_probs(target_policy_logits, actions)
    behavior_action_log_probs = action_log_probs(behavior_policy_logits, actions)
    log_rhos = target_action_log_probs - behavior_action_log_probs

    rhos = torch.exp(log_rhos)
    if clip_rho_threshold is not None:
        clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
    else:
        clipped_rhos = rhos

    cs = torch.clamp(rhos, max=1.0)
    # Append bootstrapped value to get [v1, ..., v_t+1]]

    mask = discounts > 0
    values = values * mask
    values_t_plus_1 = torch.cat(
        [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
    )
    values_t_plus_1 = values_t_plus_1 * mask
    deltas = clipped_rhos * mask * (rewards + discounts * values_t_plus_1 - values)

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
    vs_t_plus_1 = torch.cat([vs[1:], broadcasted_bootstrap_values.unsqueeze(0)], dim=0)
    if clip_pg_rho_threshold is not None:
        clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
    else:
        clipped_pg_rhos = rhos
    pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)

    # Make sure no gradients backpropagated through the returned values.
    vtrace_returns = VTraceReturns(vs=vs, pg_advantages=pg_advantages)
    return VTraceFromLogitsReturns(
        log_rhos=log_rhos,
        behavior_action_log_probs=behavior_action_log_probs,
        target_action_log_probs=target_action_log_probs,
        **vtrace_returns._asdict(),
    )


def compute_baseline_loss(advantages: torch.Tensor) -> torch.Tensor:
    return advantages**2


def compute_policy_gradient_loss(
    log_policy: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    cross_entropy = -(
        torch.flatten(log_policy, 0, 1) * torch.flatten(target, 0, 1)
    ).sum(-1)
    return cross_entropy.view(*log_policy.shape[:-1])


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

        self.extra_config = {
            "num_params": _print_params(self.params)[0],
            "entity_size": self.params.entity_size,
            "stream_size": self.params.stream_size,
        }

        self.params_actor = deepcopy(self.params).share_memory()

        self.params.train()
        self.params_actor.eval()

        self.params.to(self.config.learner_device)
        self.params_actor.to(self.config.actor_device)

        if not debug and trace_nets:
            actor_example = get_example(1, 1, device=self.config.actor_device)
            self.params_actor = torch.jit.trace(self.params_actor, actor_example)

            with torch.autocast(
                device_type=self.config.learner_device,
                dtype=torch.float16,
                enabled=self.use_amp,
            ):
                learner_example = get_example(1, 1, device=self.config.learner_device)
                self.params = torch.jit.trace(self.params, learner_example)

        # Parameter optimizers.
        self.optimizer = optim.Adam(
            self.params.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam.b1, self.config.adam.b2),
            weight_decay=self.config.weight_decay,
        )

        self.scaler = torch.cuda.amp.GradScaler()
        self.learner_steps = 0

    def save(self, fpath: str):
        torch.save(
            {
                "config": self.config,
                "params": self.params.state_dict(),
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
        obj.params_actor.load_state_dict(ckpt["params"])
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

        forward_batch = {key: self._to_torch(state[key]) for key in MODEL_INPUT_KEYS}

        heuristic_action = torch.from_numpy(state["heuristic_action"][..., -1])
        heuristic_policy = torch.eye(10)[heuristic_action]

        learner_outputs = ModelOutput(*self.params(**forward_batch))

        behavior_policy = torch.from_numpy(batch.policy)
        action = torch.from_numpy(batch.action)
        discounts = torch.from_numpy(batch.valid).to(torch.float32)
        rewards = torch.from_numpy(batch.rewards * batch.valid).squeeze(-1)

        legal = self._to_torch(batch.legal)
        values = learner_outputs.value.squeeze(-1)
        bootstrap_value = learner_outputs.value[-1].squeeze(-1)

        threshold = 0.65
        target_policy = torch.where(
            heuristic_policy.to(self.config.learner_device) > 0,
            threshold,
            legal
            * (1 - threshold)
            * torch.ones_like(learner_outputs.log_policy).detach()
            / (legal.sum(-1, keepdim=True) - 1).clamp(min=1),
        )
        target_policy = target_policy / target_policy.sum(-1, keepdim=True)

        assert torch.all(torch.abs(rewards.sum(0)) == 1)

        with torch.no_grad():
            target_values = learner_outputs.value.cpu().squeeze(-1)
            vtrace_returns = from_logits(
                behavior_policy_logits=behavior_policy,
                target_policy_logits=target_policy.cpu(),
                actions=action,
                discounts=discounts,
                rewards=rewards,
                values=target_values,
                bootstrap_value=bootstrap_value.cpu(),
            )

        discounts = discounts.to(self.config.learner_device)
        vs = vtrace_returns.vs.to(self.config.learner_device)
        action = action.to(self.config.learner_device)

        pg_loss = compute_policy_gradient_loss(
            learner_outputs.log_policy, target_policy
        )

        with torch.no_grad():
            baseline_loss = compute_baseline_loss(vs - values)
            entropy_loss = get_loss_entropy(
                learner_outputs.policy,
                learner_outputs.log_policy,
                forward_batch["legal"],
            )

        discounts_sum = discounts.sum()
        loss = (
            pg_loss
            + baseline_loss
            # + 1e-2 * (entropy_loss + torch.norm(learner_outputs.logits, dim=-1))
        )
        loss = (loss * discounts).sum()
        loss = loss / discounts_sum

        self.scaler.scale(loss).backward()

        return {
            "v_loss": ((baseline_loss * discounts).sum() / discounts_sum).item(),
            "p_loss": ((pg_loss * discounts).sum() / discounts_sum).item(),
            "e_loss": ((entropy_loss * discounts).sum() / discounts_sum).item(),
        }

    def update_parameters(self, batch: Batch):
        """A jitted pure-functional part of the `step`."""

        loss_vals = self.loss(batch)

        self.scaler.unscale_(self.optimizer)

        nn.utils.clip_grad.clip_grad_value_(
            self.params.parameters(), self.config.clip_gradient
        )

        # Update `params`` using the computed gradient.
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.optimizer.zero_grad(set_to_none=True)

        self.params_actor.load_state_dict(self.params.state_dict())

        return loss_vals
