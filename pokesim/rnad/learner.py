import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy
from dataclasses import asdict
from typing import Any, List, Mapping, Sequence

from pokesim.data import MODEL_INPUT_KEYS, NUM_PLAYERS
from pokesim.nn.model import Model
from pokesim.structs import Batch, State

from pokesim.rnad.utils import (
    EntropySchedule,
    v_trace,
    _player_others,
    optimized_forward,
)
from pokesim.rnad.config import RNaDConfig

from pokesim.utils import finetune, _print_params, SGDTowardsModel


def get_loss_v_(
    v_n: torch.Tensor,
    v_target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    loss_v = torch.unsqueeze(mask, dim=-1) * (v_n - torch.detach(v_target)) ** 2
    loss_v = torch.sum(loss_v)
    return loss_v


def get_loss_v(
    v_list: List[torch.Tensor],
    v_target_list: List[torch.Tensor],
    mask_list: List[torch.Tensor],
) -> Sequence[torch.Tensor]:
    """Define the loss function for the critic."""
    loss_v_list = []
    for v_n, v_target, mask in zip(v_list, v_target_list, mask_list):
        loss_v = get_loss_v_(v_n, v_target, mask)
        loss_v_list.append(loss_v)
    return loss_v_list


def apply_force_with_threshold(
    decision_outputs: torch.Tensor,
    force: torch.Tensor,
    threshold: float,
    threshold_center: torch.Tensor,
) -> torch.Tensor:
    """Apply the force with below a given threshold."""
    can_decrease = decision_outputs - threshold_center > -threshold
    can_increase = decision_outputs - threshold_center < threshold
    force_negative = torch.minimum(force, torch.tensor(0.0))
    force_positive = torch.maximum(force, torch.tensor(0.0))
    clipped_force = can_decrease * force_negative + can_increase * force_positive
    return decision_outputs * torch.detach(clipped_force)


def renormalize(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The `normalization` is the number of steps over which loss is computed."""
    loss = torch.sum(loss * mask)
    return loss


def get_loss_nerd_(
    logit_pi: torch.Tensor,
    pi: torch.Tensor,
    q_vr: torch.Tensor,
    valid: torch.Tensor,
    player_ids: torch.Tensor,
    legal_actions: torch.Tensor,
    is_c: torch.Tensor,
    k: int,
    clip: float = 100,
    threshold: float = 2,
) -> torch.Tensor:
    # loss policy
    adv_pi = q_vr - torch.sum(pi * q_vr, dim=-1, keepdim=True)
    adv_pi = is_c * adv_pi  # importance sampling correction
    adv_pi = torch.clip(adv_pi, min=-clip, max=clip)
    adv_pi = torch.detach(adv_pi)

    logits = logit_pi - torch.mean(logit_pi * legal_actions, dim=-1, keepdim=True)

    threshold_center = torch.zeros_like(logits)

    nerd_loss = torch.sum(
        legal_actions
        * apply_force_with_threshold(logits, adv_pi, threshold, threshold_center),
        dim=-1,
    )
    nerd_loss = -renormalize(nerd_loss, valid * (player_ids == k))

    return nerd_loss


def get_loss_nerd(
    logit_list: List[torch.Tensor],
    policy_list: List[torch.Tensor],
    q_vr_list: List[torch.Tensor],
    valid: torch.Tensor,
    player_ids: torch.Tensor,
    legal_actions: torch.Tensor,
    importance_sampling_correction: List[torch.Tensor],
    clip: float = 100,
    threshold: float = 2,
) -> Sequence[torch.Tensor]:
    """Define the nerd loss."""
    loss_pi_list = []
    for k, (logit_pi, pi, q_vr, is_c) in enumerate(
        zip(logit_list, policy_list, q_vr_list, importance_sampling_correction)
    ):
        nerd_loss = get_loss_nerd_(
            logit_pi,
            pi,
            q_vr,
            valid,
            player_ids,
            legal_actions,
            is_c,
            k,
            clip,
            threshold,
        )
        loss_pi_list.append(nerd_loss)
    return loss_pi_list


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
        config: RNaDConfig = RNaDConfig(),
        init: Mapping[str, Any] = None,
        use_amp: bool = False,
        trace_nets: bool = True,
        debug: bool = False,
    ):
        self.config = config
        self.use_amp = use_amp
        self._entropy_schedule = EntropySchedule(
            sizes=self.config.entropy_schedule_size,
            repeats=self.config.entropy_schedule_repeats,
        )

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

        self.params_target = deepcopy(self.params)
        self.params_prev = deepcopy(self.params)
        self.params_prev_ = deepcopy(self.params)

        self.params_target.train()
        self.params_prev.train()
        self.params_prev_.train()

        self.params.to(self.config.learner_device)
        self.params_actor.to(self.config.actor_device)
        self.params_target.to(self.config.learner_device)
        self.params_prev.to(self.config.learner_device)
        self.params_prev_.to(self.config.learner_device)

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
                self.params_prev = torch.jit.trace(self.params_prev, learner_example)
                self.params_prev_ = torch.jit.trace(self.params_prev_, learner_example)

        # Parameter optimizers.
        self.optimizer = optim.Adam(
            self.params.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam.b1, self.config.adam.b2),
        )
        self.optimizer_target = SGDTowardsModel(
            self.params_target, self.params, self.config.target_network_avg
        )

        self.scaler = torch.cuda.amp.GradScaler()
        self.learner_steps = 0

        self.Bbs = config.batch_size
        self.Tbs = config.backward_batch_size // self.Bbs
        print(f"({self.Tbs}, {self.Bbs}) => {self.Bbs * self.Tbs}")

    def save(self, fpath: str):
        torch.save(
            {
                "config": self.config,
                "params": self.params.state_dict(),
                "params_actor": self.params_actor.state_dict(),
                "params_target": self.params_target.state_dict(),
                "params_prev": self.params_prev.state_dict(),
                "params_prev_": self.params_prev_.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "learner_steps": self.learner_steps,
            },
            fpath,
        )

    @classmethod
    def from_fpath(cls, fpath: str, ignore_config: bool = False):
        obj = cls()
        ckpt = torch.load(fpath, map_location="cpu")
        if not ignore_config:
            obj.config = ckpt["config"]
        obj.params.load_state_dict(ckpt["params"])
        obj.params_actor.load_state_dict(ckpt["params_actor"])
        obj.params_target.load_state_dict(ckpt["params_target"])
        obj.params_prev.load_state_dict(ckpt["params_prev"])
        obj.params_prev_.load_state_dict(ckpt["params_prev_"])
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

    def loss(self, batch: Batch, alpha: float) -> float:
        state = {
            **State(batch.state).dense(),
            "history_mask": batch.history_mask,
            "legal": batch.legal,
        }

        forward_batch = {key: torch.from_numpy(state[key]) for key in MODEL_INPUT_KEYS}

        with torch.no_grad():
            with torch.autocast(
                device_type=self.config.learner_device,
                dtype=torch.float16,
                enabled=self.use_amp,
            ):
                params = optimized_forward(self.params, forward_batch, self.config)
                params_target = optimized_forward(
                    self.params_target, forward_batch, self.config
                )
                params_prev = optimized_forward(
                    self.params_prev, forward_batch, self.config
                )
                params_prev_ = optimized_forward(
                    self.params_prev_, forward_batch, self.config
                )

            # This line creates the reward transform log(pi(a|x)/pi_reg(a|x)).
            # For the stability reasons, reward changes smoothly between iterations.
            # The mixing between old and new reward transform is a convex combination
            # parametrised by alpha.
            log_policy_reg = params.log_policy - (
                alpha * params_prev.log_policy + (1 - alpha) * params_prev_.log_policy
            )
            # log_policy_reg = torch.zeros_like(params.log_policy)

            v_target_list, has_played_list, v_trace_policy_target_list = [], [], []

            _rewards = batch.rewards.astype(np.float32)
            _v_target = params_target.value.cpu().numpy()
            _policy_pprocessed = finetune(
                params.policy.detach().cpu(), forward_batch["legal"]
            ).numpy()
            _log_policy_reg = log_policy_reg.detach().cpu().numpy()

            valid = self._to_torch(batch.valid)
            player_id = self._to_torch(batch.player_id)
            legal = self._to_torch(batch.legal)

            action_oh = np.eye(params.policy.shape[-1])[batch.action]

            for player in range(NUM_PLAYERS):
                reward = _rewards[:, :, player]  # [T, B, Player]
                v_target_, has_played, policy_target_ = v_trace(
                    _v_target,
                    batch.valid,
                    batch.player_id,
                    batch.policy,
                    _policy_pprocessed,
                    _log_policy_reg,
                    _player_others(batch.player_id, batch.valid, player),
                    action_oh,
                    reward,
                    player,
                    lambda_=1.0,
                    c=self.config.c_vtrace,
                    rho=self.config.rho,
                    eta=self.config.eta_reward_transform,
                )
                v_target_ = self._to_torch(np.array(v_target_))
                has_played = self._to_torch(np.array(has_played))
                policy_target_ = self._to_torch(np.array(policy_target_))

                v_target_list.append(v_target_)
                has_played_list.append(has_played)
                v_trace_policy_target_list.append(policy_target_)

            # policy_ratio = np.array(
            #     _policy_ratio(_policy_pprocessed, batch.policy, action_oh, batch.valid)
            # )
            # is_vector = torch.unsqueeze(self._to_torch(policy_ratio), axis=-1)
            is_vector = torch.unsqueeze(torch.ones_like(valid), axis=-1)
            importance_sampling_correction = [
                torch.clamp(is_vector, max=1)
            ] * NUM_PLAYERS

            num_valid_p1 = has_played_list[0].sum()
            num_valid_p2 = has_played_list[1].sum()

        tv_loss = 0
        tp_loss = 0
        te_loss = 0
        tr_loss = 0

        T, B, *_ = batch.valid.shape

        Tnum = math.ceil(T / self.Tbs)
        Bnum = math.ceil(B / self.Bbs)

        total_valid = batch.valid.sum().item()

        for bi in range(Bnum):
            bs, bf = self.Bbs * bi, self.Bbs * (bi + 1)

            for ti in range(Tnum):
                ts, tf = self.Tbs * ti, self.Tbs * (ti + 1)

                minibatch_valid = valid[ts:tf, bs:bf]

                # An optimization to reduce the number of padded
                # samples being processed in the backward pass.
                # Only used because batch is sorted by trajectory length
                mbs = max(
                    (self.config.batch_size - minibatch_valid.any(0).sum().item()) + bs,
                    bs,
                )
                minibatch_valid = valid[ts:tf, mbs:bf]
                minibatch_valid_sum = minibatch_valid.sum().item()

                if not minibatch_valid_sum:
                    continue

                minibatch = {
                    k: v[ts:tf, mbs:bf].to(
                        self.config.learner_device, non_blocking=True
                    )
                    for k, v in forward_batch.items()
                }
                with torch.autocast(
                    device_type=self.config.learner_device,
                    dtype=torch.float16,
                    enabled=self.use_amp,
                ):
                    pi, log_pi, logit, v, recon_loss = self.params(**minibatch)

                    loss_v = get_loss_v(
                        [v] * NUM_PLAYERS,
                        [v_target[ts:tf, mbs:bf] for v_target in v_target_list],
                        [has_played[ts:tf, mbs:bf] for has_played in has_played_list],
                    )

                    # Uses v-trace to define q-values for Nerd
                    loss_nerd = get_loss_nerd(
                        [logit] * NUM_PLAYERS,
                        [pi] * NUM_PLAYERS,
                        [vtpt[ts:tf, mbs:bf] for vtpt in v_trace_policy_target_list],
                        minibatch_valid,
                        player_id[ts:tf, mbs:bf],
                        legal[ts:tf, mbs:bf],
                        [
                            is_c[ts:tf, mbs:bf]
                            for is_c in importance_sampling_correction
                        ],
                        clip=self.config.nerd.clip,
                        threshold=self.config.nerd.beta,
                    )

                    loss = 0
                    for value_loss, policy_loss, scale in zip(
                        loss_v, loss_nerd, [num_valid_p1, num_valid_p2]
                    ):
                        loss += (value_loss + policy_loss) / scale

                    loss += (recon_loss * minibatch_valid).sum() / minibatch_valid_sum

                self.scaler.scale(loss).backward()

                tv_loss += sum(loss_v).item()
                tp_loss += sum(loss_nerd).item()
                te_loss += (minibatch_valid * (log_pi * -pi).sum(-1)).sum()
                tr_loss += (minibatch_valid * recon_loss).sum()

        tv_loss /= total_valid
        tp_loss /= total_valid
        te_loss /= total_valid
        tr_loss /= total_valid

        return {
            "v_loss": tv_loss,
            "p_loss": tp_loss,
            "e": te_loss,
            "r_loss": tr_loss,
        }

    def update_parameters(self, batch: Batch, alpha: float, update_target_net: bool):
        """A jitted pure-functional part of the `step`."""

        loss_vals = self.loss(batch, alpha)

        self.scaler.unscale_(self.optimizer)

        nn.utils.clip_grad.clip_grad_value_(
            self.params.parameters(), self.config.clip_gradient
        )

        # Update `params`` using the computed gradient.
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.optimizer.zero_grad(set_to_none=True)

        # Update `params_target` towards `params`.
        self.optimizer_target.step()

        # Rolls forward the prev and prev_ params if update_target_net is 1.
        # pyformat: disable
        if update_target_net:
            print(f"Updating regularization nets @ {self.learner_steps:,}")
            self.params_prev_.load_state_dict(self.params_prev.state_dict())
            self.params_prev.load_state_dict(self.params_target.state_dict())

        self.params_actor.load_state_dict(self.params.state_dict())

        draws = abs(batch.rewards).sum(0).sum(1) == 0
        draw_ratio = draws.sum() / batch.valid.shape[1]

        logs = {
            **loss_vals,
            "draw_ratio": draw_ratio.item(),
        }
        return logs
