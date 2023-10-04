import wandb
import random
import asyncio

import torch
import numpy as np
import multiprocessing as mp

from abc import ABC, abstractmethod

from typing import Tuple

from pokesim.utils import preprocess
from pokesim.inference import Inference
from pokesim.constants import (
    _DEFAULT_ACTION,
    _RANDOM_ACTION,
    _MAXDMG_ACTION,
    _LINE_FEED,
)
from pokesim.structs import EnvStep, ActorStep, ModelOutput


class Action:
    def __init__(self, game_id: int, player_id: int):
        self.game_id = game_id
        self.player_id = player_id

    @classmethod
    def from_env_step(cls, env_step: EnvStep):
        return cls(env_step.game_id.item(), env_step.player_id.item())

    def select_action(self, action_index: int):
        return bytearray([self.game_id, self.player_id, action_index, _LINE_FEED])

    def default(self):
        return self.select_action(_DEFAULT_ACTION)

    def random(self):
        return self.select_action(_RANDOM_ACTION)

    def maxdmg(self):
        return self.select_action(_MAXDMG_ACTION)


class Actor(ABC):
    async def _choose_action(self, env_step: EnvStep):
        try:
            return await self.choose_action(env_step)
        except Exception:
            import traceback

            traceback.print_exc()

    @abstractmethod
    async def choose_action(self, env_step: EnvStep) -> Tuple[Action, ActorStep]:
        raise NotImplementedError

    def _done_callback(self, reward: int):
        pass


def _get_action(policy: np.ndarray):
    action = random.choices(
        population=list(range(policy.shape[-1])),
        k=1,
        weights=policy.squeeze().tolist(),
    )
    return action[0]


class SelfplayActor(Actor):
    def __init__(self, inference: Inference):
        self.inference = inference
        self.loop = asyncio.get_running_loop()

    @torch.no_grad()
    def _forward_model(
        self,
        teams: torch.Tensor,
        side_conditions: torch.Tensor,
        volatile_status: torch.Tensor,
        boosts: torch.Tensor,
        field: torch.Tensor,
        mask: torch.Tensor,
        action_hist: torch.Tensor,
    ) -> ModelOutput:
        return self.inference.model(
            teams=teams,
            side_conditions=side_conditions,
            volatile_status=volatile_status,
            boosts=boosts,
            field=field,
            mask=mask,
            action_hist=action_hist,
        )

    async def choose_action(
        self, env_step: EnvStep, policy_select: int
    ) -> Tuple[Action, ActorStep]:
        batch = preprocess(env_step.raw_obs)
        batch = {k: torch.from_numpy(v) for k, v in batch.items()}
        legal = env_step.legal.astype(np.bool_)
        policy_select = np.array([[policy_select]])
        move_mask = legal[..., :4]
        switch_mask = legal[..., 4:]
        action_type_mask = np.concatenate(
            (
                move_mask.sum(-1, keepdims=True) > 0,
                switch_mask.sum(-1, keepdims=True) > 0,
            ),
            axis=-1,
        )
        move_mask += ~action_type_mask[..., 0][..., None]
        switch_mask += ~action_type_mask[..., 1][..., None]
        policy_select = policy_select[..., None]
        total_mask = np.concatenate(
            (
                action_type_mask * (policy_select == 0),
                move_mask * (policy_select == 1),
                switch_mask * (policy_select == 2),
            ),
            axis=-1,
        )
        mask = torch.from_numpy(total_mask)
        model_output = self._forward_model(**batch, mask=mask)
        policy = model_output.policy.detach().numpy()
        action = _get_action(policy)
        return (
            Action.from_env_step(env_step).select_action(action - 2)
            if policy_select > 0
            else None,
            ActorStep(
                policy=policy,
                action=np.array([[action]]),
                policy_select=policy_select,
            ),
        )


class EvalActor(Actor):
    def __init__(self, pi_type: str, eval_queue: mp.Queue) -> None:
        self.pi_type = pi_type
        self.eval_queue = eval_queue
        self.n = 0

    def _done_callback(self, reward: int):
        self.eval_queue.put((self.n, self.pi_type, reward))
        self.n += 1


class DefaultEvalActor(EvalActor):
    async def choose_action(self, env_step: EnvStep) -> Tuple[Action, ActorStep]:
        return (Action.from_env_step(env_step).default(), None)


class RandomEvalActor(EvalActor):
    async def choose_action(self, env_step: EnvStep) -> Tuple[Action, ActorStep]:
        batch = preprocess(env_step.raw_obs)
        batch = {k: torch.from_numpy(v) for k, v in batch.items()}
        mask = torch.from_numpy(env_step.legal.astype(np.bool_))
        policy = (
            torch.masked_fill(torch.ones(10), ~mask, float("-inf")).softmax(-1).numpy()
        )
        action = _get_action(policy)
        return (Action.from_env_step(env_step).select_action(action), None)


class MaxdmgEvalActor(EvalActor):
    async def choose_action(self, env_step: EnvStep) -> Tuple[Action, ActorStep]:
        return (Action.from_env_step(env_step).maxdmg(), None)
