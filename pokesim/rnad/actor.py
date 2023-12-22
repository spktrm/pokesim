import asyncio
import torch
import random

import numpy as np
import torch.nn as nn
import multiprocessing as mp

from typing import Dict, List, Tuple
from pokesim.data import ENCODING, EVAL_WORKER_INDEX, MODEL_INPUT_KEYS, ACTION_SPACE
from pokesim.env import EnvironmentNoStackSingleStep
from pokesim.structs import (
    ActorStep,
    EnvStep,
    ModelOutput,
    Observation,
    TimeStep,
)

from pokesim.structs import Trajectory
from pokesim.utils import finetune


def handle_verbose(
    n: int, pi: np.ndarray, logit: np.ndarray, action: np.ndarray, value: np.ndarray
):
    nice_probs = [f"{p:.2f}" for p in pi]
    action_type = " ".join(nice_probs)

    nice_logit = [f"{p:.2f}" for p in logit]
    logits = " ".join(nice_logit)

    v = f"{value.item():.3f}"

    text = "\n".join([str(n), action_type, logits, f"{action}", v])
    print(text + "\n")


_MODEL_INPUT_KEYS = MODEL_INPUT_KEYS.copy()
_MODEL_INPUT_KEYS.remove("history_mask")


def _actor_step(
    worker_index: int,
    model: nn.Module,
    obs: Dict[str, torch.Tensor],
    verbose: bool = False,
) -> ActorStep:
    model_input = {key: torch.from_numpy(obs[key]) for key in _MODEL_INPUT_KEYS}
    model_input["legal"] = model_input["legal"].reshape(1, 1, -1)

    actor_model = model
    with torch.no_grad():
        model_output: ModelOutput = actor_model(**model_input)
    logit = model_output.logits
    pi = model_output.policy
    value = model_output.value

    logit = logit.cpu().numpy().flatten()

    if worker_index >= EVAL_WORKER_INDEX:
        pi = finetune(pi, model_input["legal"])
    pi = pi.cpu().numpy().flatten()

    value = value.cpu().numpy().flatten()

    action = random.choices(ACTION_SPACE, weights=pi)[0]

    if verbose:
        handle_verbose(0, pi, logit, action, value)

    return ActorStep(policy=pi, action=action, value=value, rewards=())


async def _apply_action(env: EnvironmentNoStackSingleStep, action: int):
    return await env.step(action)


def _state_as_env_step(worker_index, obs, rewards, done, player_index) -> EnvStep:
    return EnvStep(
        game_id=worker_index,
        player_id=player_index,
        state=obs["raw"],
        rewards=rewards,
        valid=not done,
        legal=obs["legal"],
    )


def _reset(
    worker_index: int,
    learn_queue: mp.Queue,
    eval_queue: mp.Queue,
    num_battles: int,
    timesteps: List[TimeStep],
):
    if worker_index < EVAL_WORKER_INDEX:
        if timesteps:
            trajectory = Trajectory.from_env_steps(timesteps)
            learn_queue.put(trajectory.serialize())

    else:
        if timesteps:
            final_reward = timesteps[-1].actor.rewards[0].item()
            eval_queue.put((num_battles, worker_index, final_reward))


async def _drain(player_index: int, env: EnvironmentNoStackSingleStep):
    action = f"{player_index}|{0}\n"
    env.writer.write(action.encode(ENCODING))
    await env.writer.drain()


async def _run_environment_async(
    worker_index: int,
    model: nn.Module,
    model_prev: nn.Module,
    learn_queue: mp.Queue,
    eval_queue: mp.Queue,
    condition: mp.Condition,
    verbose: bool = False,
    threshold: int = 2,
):
    timesteps: List[TimeStep] = []

    num_battles = 0

    env = await EnvironmentNoStackSingleStep.create(worker_index, threshold=threshold)

    while True:
        dones = 0
        num_steps = 0

        obs, reward, done, player_index = await env.reset()
        env_step = _state_as_env_step(worker_index, obs, reward, done, player_index)

        while True:
            prev_env_step = env_step
            actor_step = _actor_step(worker_index, model, obs, verbose)

            obs, reward, done, player_index = await _apply_action(
                env, actor_step.action
            )
            env_step = _state_as_env_step(worker_index, obs, reward, done, player_index)
            dones += done
            rewards = np.zeros((2,))
            if dones >= 2:
                rewards[player_index] = reward
                rewards[1 - player_index] = -reward
            timesteps.append(
                TimeStep(
                    id=f"{worker_index}{player_index}",
                    env=prev_env_step,
                    actor=ActorStep(
                        action=actor_step.action,
                        policy=actor_step.policy,
                        rewards=rewards,
                        value=actor_step.value,
                    ),
                    ts=num_steps,
                )
            )
            num_steps += 1
            if dones >= 2:
                _reset(worker_index, learn_queue, eval_queue, num_battles, timesteps)
                timesteps = []
                await _drain(player_index, env)
                break

        num_battles += 1


def run_environment(
    worker_index: int,
    model: nn.Module,
    model_prev: nn.Module,
    learn_queue: mp.Queue,
    eval_queue: mp.Queue,
    condition: mp.Condition,
    verbose: bool = False,
    threshold: int = 2,
):
    return asyncio.run(
        _run_environment_async(
            worker_index,
            model,
            model_prev,
            learn_queue,
            eval_queue,
            condition,
            verbose,
            threshold,
        )
    )
