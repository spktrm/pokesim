import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from copy import deepcopy
from dataclasses import asdict
from typing import Any, List, Mapping

from pokesim.data import MODEL_INPUT_KEYS, NUM_HISTORY, NUM_PLAYERS
from pokesim.nn.model import Model
from pokesim.structs import Batch, ModelOutput, State

from pokesim.rnad.utils import EntropySchedule, v_trace, _player_others, _policy_ratio
from pokesim.rnad.config import RNaDConfig

from pokesim.utils import finetune, _print_params, SGDTowardsModel, get_example


def get_loss_v(
    v_list: List[torch.Tensor],
    v_target_list: List[torch.Tensor],
    mask_list: List[torch.Tensor],
) -> torch.Tensor:
    """Define the loss function for the critic."""
    loss_v_list = []
    for v_n, v_target, mask in zip(v_list, v_target_list, mask_list):
        loss_v = torch.unsqueeze(mask, dim=-1) * (v_n - v_target.detach()) ** 2
        normalization = torch.sum(mask)
        loss_v = torch.sum(loss_v) / (normalization + (normalization == 0.0))
        loss_v_list.append(loss_v)
    return sum(loss_v_list)


def apply_force_with_threshold(
    decision_outputs: torch.Tensor,
    force: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Apply the force with below a given threshold."""
    can_decrease = decision_outputs > -threshold
    can_increase = decision_outputs < threshold
    force_negative = torch.minimum(force, torch.tensor(0.0))
    force_positive = torch.maximum(force, torch.tensor(0.0))
    clipped_force = can_decrease * force_negative + can_increase * force_positive
    return decision_outputs * torch.detach(clipped_force)


def renormalize(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The `normalization` is the number of steps over which loss is computed."""
    loss = torch.sum(loss * mask)
    normalization = torch.sum(mask)
    return loss / (normalization + (normalization == 0.0))


def get_loss_entropy(
    policy: torch.Tensor, log_policy: torch.Tensor, legal: torch.Tensor
) -> torch.Tensor:
    loss_entropy = (policy * log_policy).sum(-1)
    num_legal_actions = legal.sum(-1)
    denom = torch.log(num_legal_actions)
    denom = torch.where(num_legal_actions <= 1, 1, denom)
    loss_entropy = loss_entropy / denom
    return loss_entropy


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
) -> torch.Tensor:
    """Define the nerd loss."""
    loss_pi_list = []

    legal_action_sum = legal_actions.sum(-1, keepdim=True)

    for k, (logit_pi, pi, q_vr, is_c) in enumerate(
        zip(logit_list, policy_list, q_vr_list, importance_sampling_correction)
    ):
        # loss policy
        adv_pi = q_vr - torch.sum(pi * q_vr, dim=-1, keepdim=True)
        adv_pi = is_c * adv_pi  # importance sampling correction
        adv_pi = torch.clip(adv_pi, min=-clip, max=clip).detach()

        valid_logit_sum = torch.sum(logit_pi * legal_actions, dim=-1, keepdim=True)
        logit_mean = valid_logit_sum / legal_action_sum
        logits = logit_pi - logit_mean

        nerd_loss = torch.sum(
            legal_actions * apply_force_with_threshold(logits, adv_pi, threshold),
            dim=-1,
        ) / legal_action_sum.squeeze(-1)

        nerd_loss = -renormalize(nerd_loss, valid * (player_ids == k))
        loss_pi_list.append(nerd_loss)
    return sum(loss_pi_list)


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
            sizes=[
                s * self.config.accum_steps for s in self.config.entropy_schedule_size
            ],
            repeats=self.config.entropy_schedule_repeats,
        )

        # param_keys = list(init.keys())
        # for key in param_keys:
        #     if key.startswith("action") or key.startswith("value"):
        #         init.pop(key)

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
        self.params_target = deepcopy(self.params)

        self.params.train()
        self.params_actor.eval()
        self.params_target.train()

        if self.config.enable_regularization:
            self.params_prev = deepcopy(self.params)
            self.params_prev_ = deepcopy(self.params)

            self.params_prev.train()
            self.params_prev_.train()

        self.params.to(self.config.learner_device)
        self.params_actor.to(self.config.actor_device)
        self.params_target.to(self.config.learner_device)

        if self.config.enable_regularization:
            self.params_prev.to(self.config.learner_device)
            self.params_prev_.to(self.config.learner_device)

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
                self.params_target = torch.jit.trace(
                    self.params_target, learner_example
                )
                if self.config.enable_regularization:
                    self.params_prev = torch.jit.trace(
                        self.params_prev, learner_example
                    )
                    self.params_prev_ = torch.jit.trace(
                        self.params_prev_, learner_example
                    )

        # Parameter optimizers.
        self.optimizer = optim.Adam(
            self.params.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam.b1, self.config.adam.b2),
        )
        self.optimizer_target = SGDTowardsModel(
            self.params_target, self.params, self.config.target_network_avg
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=False)
        self.learner_steps = 0
        self.scale_factor = 0

    def save(self, fpath: str):
        obj = {
            "config": self.config,
            "params": self.params.state_dict(),
            "params_actor": self.params_actor.state_dict(),
            "params_target": self.params_target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "learner_steps": self.learner_steps,
        }
        if self.config.enable_regularization:
            obj["params_prev"] = self.params_prev.state_dict()
            obj["params_prev_"] = self.params_prev_.state_dict()
        torch.save(obj, fpath)

    @classmethod
    def from_fpath(cls, fpath: str, ignore_config: bool = False, **kwargs):
        obj = cls(**kwargs)
        ckpt = torch.load(fpath, map_location="cpu")
        if not ignore_config:
            obj.config = ckpt["config"]
        params = ckpt["params"]
        obj.params.load_state_dict(ckpt["params"])
        obj.params_actor.load_state_dict(ckpt.get("params_actor", params))
        obj.params_target.load_state_dict(ckpt["params_target"])
        obj.optimizer.load_state_dict(ckpt["optimizer"])
        obj.scaler.load_state_dict(ckpt["scaler"])
        obj.learner_steps = ckpt["learner_steps"]
        if obj.config.enable_regularization:
            try:
                obj.params_prev.load_state_dict(ckpt["params_prev"])
            except:
                obj.params_prev.load_state_dict(ckpt["params_target"])
            try:
                obj.params_prev_.load_state_dict(ckpt["params_prev_"])
            except:
                obj.params_prev_.load_state_dict(ckpt["params_target"])
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
            "legal": batch.legal,
        }

        forward_batch = {key: self._to_torch(state[key]) for key in MODEL_INPUT_KEYS}

        assert np.all(
            (np.where(batch.valid, batch.game_id, -1) == batch.game_id[0, None]).mean(0)
            == batch.valid.mean(0)
        )

        params = ModelOutput(*self.params(**forward_batch))

        with torch.no_grad():
            params_target = ModelOutput(*self.params_target(**forward_batch))

            if self.config.enable_regularization:
                params_prev = ModelOutput(*self.params_prev(**forward_batch))
                params_prev_ = ModelOutput(*self.params_prev_(**forward_batch))

        # This line creates the reward transform log(pi(a|x)/pi_reg(a|x)).
        # For the stability reasons, reward changes smoothly between iterations.
        # The mixing between old and new reward transform is a convex combination
        # parametrised by alpha.
        if self.config.enable_regularization:
            log_policy_reg = params.log_policy - (
                alpha * params_prev.log_policy + (1 - alpha) * params_prev_.log_policy
            )
        else:
            log_policy_reg = torch.zeros_like(params.log_policy)

        v_target_list, has_played_list, v_trace_policy_target_list = [], [], []

        _rewards = batch.rewards.astype(np.float32)
        _v_target = params_target.value.cpu().numpy()

        # _policy_pprocessed = np.eye(10)[state["heuristic_action"][..., -1]]
        _policy_pprocessed = finetune(
            params.policy.detach().cpu(), torch.from_numpy(batch.legal)
        ).numpy()
        # _policy_pprocessed = params.policy.detach().cpu().numpy()

        _log_policy_reg = log_policy_reg.detach().cpu().numpy()
        valid = self._to_torch(batch.valid)
        player_id = self._to_torch(batch.player_id)
        legal = self._to_torch(batch.legal)

        action_oh = np.eye(params.policy.shape[-1])[batch.action]

        assert np.all(np.abs(_rewards.sum(0)) == 1)

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

        loss_v = get_loss_v(
            [params.value] * NUM_PLAYERS, v_target_list, has_played_list
        )

        policy_ratio = np.array(
            _policy_ratio(_policy_pprocessed, batch.policy, action_oh, batch.valid)
        )
        is_vector = torch.unsqueeze(self._to_torch(policy_ratio), axis=-1)
        is_vector = is_vector.clamp(max=1)
        # is_vector = torch.unsqueeze(torch.ones_like(valid), axis=-1)
        importance_sampling_correction = [is_vector] * NUM_PLAYERS

        # Uses v-trace to define q-values for Nerd
        loss_nerd = get_loss_nerd(
            [params.logits] * NUM_PLAYERS,
            [params.policy] * NUM_PLAYERS,
            v_trace_policy_target_list,
            valid,
            player_id,
            legal,
            importance_sampling_correction,
            clip=self.config.nerd.clip,
            threshold=self.config.nerd.beta,
        )

        valid_sum = valid.sum()

        # heuristic_action = torch.from_numpy(state["heuristic_action"][..., -1])
        # heuristic_action = heuristic_action.to(self.config.learner_device)
        # heurisitc_loss = F.cross_entropy(
        #     torch.flatten(params.logits, 0, 1),
        #     torch.flatten(heuristic_action, 0, 1),
        #     reduction="none",
        # ).view_as(heuristic_action)
        # heurisitc_loss = (heurisitc_loss * valid).sum() / valid_sum

        loss = (
            loss_v
            + loss_nerd
            # + max(0, (1 - self.learner_steps / 10000)) * heurisitc_loss
        )

        if not self.config.enable_regularization:
            loss_entropy = get_loss_entropy(params.policy, params.log_policy, legal)
            loss_entropy = (loss_entropy * valid).sum() / valid_sum
            # loss = loss + 1e-3 * loss_entropy

        else:
            with torch.no_grad():
                loss_entropy = get_loss_entropy(params.policy, params.log_policy, legal)
                loss_entropy = (loss_entropy * valid).sum() / valid_sum

        # loss = loss / self.config.accum_steps

        self.scaler.scale(loss).backward()

        return {
            "v_loss": loss_v.item(),
            "p_loss": loss_nerd.item(),
            "e_loss": loss_entropy.item(),
        }

    def update_parameters(self, batch: Batch, alpha: float, update_target_net: bool):
        """A jitted pure-functional part of the `step`."""

        loss_vals = self.loss(batch, alpha)
        self.scale_factor += batch.valid.sum()

        if (
            self.learner_steps % self.config.accum_steps == 0
        ) and self.learner_steps > 0:
            self.scaler.unscale_(self.optimizer)

            # for param in self.params.parameters():
            #     if param.grad is not None:
            #         param.grad.data /= self.scale_factor
            self.scale_factor = 0

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
            if update_target_net and self.config.enable_regularization:
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
