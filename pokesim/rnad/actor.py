import asyncio
import torch
import random

import numpy as np
import torch.nn as nn
import multiprocessing as mp

from typing import Dict, List, Tuple
from pokesim.data import EVAL_WORKER_INDEX, MODEL_INPUT_KEYS, PREV_WORKER_INDEX
from pokesim.env import EnvironmentNoStackSingleStep
from pokesim.structs import (
    ActorStep,
    EnvStep,
    ModelOutput,
    TimeStep,
)

from pokesim.structs import Trajectory


def handle_verbose(n: int, pi: np.ndarray, action: np.ndarray, value: np.ndarray):
    nice_probs = [f"{p:.2f}" for p in pi]
    action_type = " ".join(nice_probs)
    v = f"{value.item():.3f}"

    text = "\n".join(
        [
            str(n),
            action_type,
            f"{action}",
            v,
        ]
    )
    print(text + "\n")


_MODEL_INPUT_KEYS = MODEL_INPUT_KEYS.copy()
_MODEL_INPUT_KEYS.remove("history_mask")


async def _run_environment_async(
    worker_index: int,
    model: nn.Module,
    model_prev: nn.Module,
    learn_queue: mp.Queue,
    eval_queue: mp.Queue,
    verbose: bool = False,
):
    timesteps: Dict[int, List[TimeStep]] = {
        0: [],
        1: [],
    }
    hidden_states = {
        0: model.get_hidden_state(1),
        1: model.get_hidden_state(1),
    }

    action_space = list(range(10))

    num_battles = 0
    num_steps = 0

    def _act(obs, reward, done, player_index):
        nonlocal num_steps

        model_input = {key: torch.from_numpy(obs[key]) for key in _MODEL_INPUT_KEYS}
        model_input["legal"] = model_input["legal"].reshape(1, 1, -1)

        if worker_index == PREV_WORKER_INDEX and player_index == 1:
            actor_model = model_prev
        else:
            actor_model = model

        with torch.no_grad():
            model_output: ModelOutput = actor_model(
                **model_input, hidden_state=hidden_states[player_index]
            )

        pi = model_output.policy
        value = model_output.value
        hidden_states[player_index] = model_output.hidden_state

        pi = pi.cpu().numpy().flatten()
        value = value.cpu().numpy().flatten()
        action = random.choices(action_space, weights=pi)[0]

        if (obs["legal"].sum() > 1) or done:
            env_step = EnvStep(
                game_id=worker_index,
                player_id=player_index,
                state=obs["raw"],
                rewards=reward,
                valid=done,
                legal=obs["legal"],
            )

            actor_step = ActorStep(
                policy=pi,
                action=action,
                rewards=reward,
                value=value,
            )

            timestep = TimeStep(id="", actor=actor_step, env=env_step, ts=num_steps)
            timesteps[env_step.player_id].append(timestep)
            num_steps += 1

        if verbose:
            handle_verbose(0, pi, action, value)

        return action

    def _reset():
        nonlocal num_battles, num_steps

        if worker_index < EVAL_WORKER_INDEX:
            trajectory_unsorted = timesteps[0] + timesteps[1]
            if trajectory_unsorted:
                trajectory1 = Trajectory.from_env_steps(timesteps[0], fix_rewards=False)
                trajectory2 = Trajectory.from_env_steps(timesteps[1], fix_rewards=False)
                learn_queue.put((trajectory1.serialize(), trajectory2.serialize()))

        else:
            if timesteps[0]:
                final_reward = timesteps[0][-1].actor.rewards.item()
                eval_queue.put((num_battles, worker_index, final_reward))

        for player_index in range(2):
            hidden_states[player_index] = model.get_hidden_state(1)
            timesteps[player_index] = []

        num_battles += 1
        num_steps = 0

    env = await EnvironmentNoStackSingleStep.create(worker_index, _act, _reset)
    await env.run()


def run_environment(
    worker_index: int,
    model: nn.Module,
    model_prev: nn.Module,
    learn_queue: mp.Queue,
    eval_queue: mp.Queue,
    verbose: bool = False,
):
    return asyncio.run(
        _run_environment_async(
            worker_index, model, model_prev, learn_queue, eval_queue, verbose
        )
    )
