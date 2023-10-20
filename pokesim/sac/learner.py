import inspect
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy
from typing import Any, Mapping

from pokesim.data import MODEL_INPUT_KEYS
from pokesim.model import Model
from pokesim.structs import Batch, State

from pokesim.sac.utils import gae_advantages
from pokesim.sac.config import SACConfig
from pokesim.utils import _print_params, SGDTowardsModel


def get_example(T: int, B: int, device: str):
    mask = torch.zeros(T, B, 12, dtype=torch.bool, device=device)
    mask[..., 0] = True
    return (
        torch.zeros(T, B, 8, dtype=torch.long, device=device),
        torch.zeros(T, B, 4, dtype=torch.long, device=device),
        torch.zeros(T, B, 8, 3, 6, 11, dtype=torch.long, device=device),
        torch.zeros(T, B, 8, 2, 15, dtype=torch.long, device=device),
        torch.zeros(T, B, 8, 2, 20, 2, dtype=torch.long, device=device),
        torch.zeros(T, B, 8, 2, 7, dtype=torch.long, device=device),
        torch.zeros(T, B, 8, 5, 3, dtype=torch.long, device=device),
        mask,
        torch.ones(T, B, 8, 4, 3, dtype=torch.long, device=device),
        torch.ones(T, B, 8, dtype=torch.bool, device=device),
    )


class LogAlpha(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.value = nn.Parameter(torch.tensor([0.0]))

    def forward(self):
        return


class Learner:
    def __init__(
        self,
        init: Mapping[str, Any] = None,
        config: SACConfig = SACConfig(),
        use_amp: bool = False,
        trace_nets: bool = True,
        debug: bool = False,
    ):
        self.config = config
        self.use_amp = use_amp

        # Create initial parameters.
        self.params = Model()

        _print_params(self.params)

        if init is not None:
            self.params.load_state_dict(init, strict=False)
        self.params_actor = deepcopy(self.params).share_memory()
        self.params_actor_prev = deepcopy(self.params).share_memory()
        self.params_target = deepcopy(self.params)

        self.params.to(self.config.learner_device)
        self.params_actor.to(self.config.actor_device)
        self.params_target.to(self.config.learner_device)

        if not debug and trace_nets:
            actor_example = get_example(1, 1, self.config.actor_device)
            self.params_actor = torch.jit.trace(self.params_actor, actor_example)
            self.params_actor_prev = torch.jit.trace(
                self.params_actor_prev, actor_example
            )
            with torch.autocast(
                device_type=self.config.learner_device,
                dtype=torch.float16,
                enabled=self.use_amp,
            ):
                learner_example = get_example(1, 1, self.config.learner_device)

                self.params = torch.jit.trace(self.params, learner_example)
                self.params_target = torch.jit.trace(
                    self.params_target, learner_example
                )

        # Parameter optimizers.
        self.optimizer = optim.Adam(
            self.params.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam.b1, self.config.adam.b2),
        )
        self.optimizer_target = SGDTowardsModel(
            self.params_target, self.params, self.config.tau
        )

        self.target_entropy = -12  # -dim(A)

        self.log_alpha = LogAlpha()
        self.log_alpha = self.log_alpha.to(self.config.learner_device)
        self.alpha_optimizer = optim.Adam(
            params=self.log_alpha.parameters(), lr=self.config.learning_rate
        )

        self.scaler = torch.cuda.amp.GradScaler()
        self.learner_steps = 0

        self.Bbs = config.batch_size
        self.Tbs = config.backward_batch_size // self.Bbs
        print(f"({self.Tbs}, {self.Bbs}) => {self.Bbs * self.Tbs}")

    def get_config(self):
        return {
            **{
                k: v.__dict__ if inspect.isclass(v) else v
                for k, v in self.config.__dict__.items()
            }
        }

    def _to_torch(self, arr: np.ndarray, device: str = None):
        if device is None:
            device = self.config.learner_device
        return torch.from_numpy(arr).to(device, non_blocking=True)

    def loss(self, batch: Batch) -> float:
        state = {
            **State(batch.state).dense(),
            "history_mask": batch.history_mask,
            "legal": batch.legal,
        }

        forward_batch = {key: self._to_torch(state[key]) for key in MODEL_INPUT_KEYS}

        with torch.autocast(
            device_type=self.config.learner_device,
            dtype=torch.float16,
            enabled=self.use_amp,
        ):
            (policy, log_policy, _, value) = self.params(**forward_batch)

            with torch.no_grad():
                (
                    policy_target,
                    log_policy_target,
                    _,
                    value_target,
                ) = self.params_target(**forward_batch)

        scale = batch.valid.sum().item()

        alpha = torch.exp(self.log_alpha.value)
        valid = self._to_torch(batch.valid)
        rewards = self._to_torch(batch.rewards)

        tp_loss = (
            valid * (policy * (alpha.detach() * log_policy - value)).sum(-1)
        ).sum()

        entropy = (policy * log_policy).sum(-1)
        te_loss = (valid * -(alpha * (entropy + self.target_entropy).detach())).sum()

        value_target_next = policy.detach() * (
            value_target - alpha * log_policy.detach()
        )

        # Compute Q targets for current states (y_i)
        value_targets = rewards + (
            self.config.gamma * valid * value_target_next.sum(-1)
        )

        tv_loss = (valid * torch.square(value - value_targets.detach())).sum()

        total_loss = (tv_loss + te_loss + tp_loss) / scale

        self.scaler.scale(total_loss).backward()

        return {
            "v_loss": tv_loss.item() / scale,
            "e_loss": te_loss.item() / scale,
            "p_loss": tp_loss.item() / scale,
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

        self.optimizer_target.step()

        self.optimizer.zero_grad(set_to_none=True)

        self.params_actor.load_state_dict(self.params.state_dict())

        if self.learner_steps % 1000 == 0:
            self.params_actor_prev.load_state_dict(self.params_actor.state_dict())

        return loss_vals
