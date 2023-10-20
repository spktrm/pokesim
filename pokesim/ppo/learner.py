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

from pokesim.ppo.utils import gae_advantages
from pokesim.ppo.config import PPOConfig
from pokesim.utils import _print_params


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


class Learner:
    def __init__(
        self,
        init: Mapping[str, Any] = None,
        config: PPOConfig = PPOConfig(),
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

        self.params.to(self.config.learner_device)
        self.params_actor.to(self.config.actor_device)

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

        # Parameter optimizers.
        self.optimizer = optim.Adam(
            self.params.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam.b1, self.config.adam.b2),
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

        values = value.detach().cpu().squeeze(-1).numpy()
        advantages = gae_advantages(
            batch.rewards[1:],
            batch.valid,
            values,
            self.config.gamma,
            self.config.gae_param,
        )
        returns = self._to_torch(advantages - batch.value.squeeze(-1)[:-1])

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = self._to_torch(advantages)

        valid = self._to_torch(batch.valid)
        scale = batch.valid.sum()

        tv_loss = torch.square(valid[:-1] * (returns - value[:-1].squeeze())).sum()
        te_loss = (valid * (-policy * log_policy).sum(-1)).sum()

        actions = self._to_torch(batch.action).unsqueeze(-1)
        old_policy = self._to_torch(batch.policy)
        policy_ratio = (
            torch.gather(policy, -1, actions) / torch.gather(old_policy, -1, actions)
        ).squeeze(-1)
        pg_loss = policy_ratio[:-1] * advantages
        clipped_pg_loss = (
            torch.clamp(
                policy_ratio[-1],
                min=1 - self.config.clip_param,
                max=1 + self.config.clip_param,
            )
            * advantages
        )
        ppo_loss = -torch.min(pg_loss, clipped_pg_loss)

        tp_loss = (valid[:-1] * ppo_loss).sum()

        total_loss = (
            self.config.vf_coeff * tv_loss
            + self.config.entropy_coeff * te_loss
            + tp_loss
        ) / scale

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

        self.optimizer.zero_grad(set_to_none=True)

        self.params_actor.load_state_dict(self.params.state_dict())

        if self.learner_steps % 1000 == 0:
            self.params_actor_prev.load_state_dict(self.params_actor.state_dict())

        return loss_vals
