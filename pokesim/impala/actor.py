import asyncio
import torch
import random

import numpy as np
import torch.nn as nn
import torch.multiprocessing as mp

from typing import Dict, List
from pokesim.data import ENCODING, EVAL_WORKER_INDEX, MODEL_INPUT_KEYS, ACTION_SPACE
from pokesim.env import EnvironmentNoStackSingleStep
from pokesim.structs import ActorStep, EnvStep, ModelOutput, TimeStep

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


def _actor_step(
    worker_index: int,
    model: nn.Module,
    obs: Dict[str, torch.Tensor],
    verbose: bool = False,
) -> ActorStep:
    model_input = {key: torch.from_numpy(obs[key]) for key in _MODEL_INPUT_KEYS}
    model_input["legal"] = model_input["legal"].reshape(1, 1, -1)

    with torch.no_grad():
        model_output = ModelOutput(*model(**model_input))
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
    timesteps: Dict[int, List[TimeStep]],
):
    if worker_index < EVAL_WORKER_INDEX:
        for player_index in range(2):
            if timesteps[player_index]:
                last_timestep = timesteps[player_index][-1]
                second_last_timestep = timesteps[player_index][-2]
                timesteps[player_index][-2] = TimeStep(
                    id=second_last_timestep.id,
                    env=second_last_timestep.env,
                    actor=ActorStep(
                        action=second_last_timestep.actor.action,
                        policy=second_last_timestep.actor.policy,
                        rewards=last_timestep.actor.rewards,
                        value=second_last_timestep.actor.value,
                    ),
                    ts=second_last_timestep.ts,
                )
                trajectory = Trajectory.from_env_steps(timesteps[player_index][:-1])
                learn_queue.put(trajectory.serialize())

    else:
        if timesteps[0]:
            final_reward = timesteps[0][-1].actor.rewards
            eval_queue.put((num_battles, worker_index, final_reward))


async def _drain(player_index: int, env: EnvironmentNoStackSingleStep):
    action = f"{player_index}|{0}\n"
    env.writer.write(action.encode(ENCODING))
    await env.writer.drain()


async def _run_environment_async(
    worker_index: int,
    model: nn.Module,
    learn_queue: mp.Queue,
    eval_queue: mp.Queue,
    verbose: bool = False,
    threshold: int = 1,
):
    timesteps: Dict[int, List[TimeStep]] = {0: [], 1: []}

    num_battles = 0

    env = await EnvironmentNoStackSingleStep.create(worker_index, threshold=threshold)

    while True:
        dones = 0
        num_steps = 0

        sid, obs, reward, done, player_index = await env.reset()
        env_step = _state_as_env_step(worker_index, obs, reward, done, player_index)

        while True:
            prev_env_step = env_step
            actor_step = _actor_step(worker_index, model, obs, verbose)

            sid, obs, reward, done, player_index = await _apply_action(
                env, actor_step.action
            )
            env_step = _state_as_env_step(worker_index, obs, reward, done, player_index)
            dones += done
            rewards = np.zeros(2)
            rewards[player_index] = reward
            rewards[1 - player_index] = -reward
            timesteps[prev_env_step.player_id].append(
                TimeStep(
                    id=f"{sid}{prev_env_step.player_id}",
                    env=prev_env_step,
                    actor=ActorStep(
                        action=actor_step.action,
                        policy=actor_step.policy,
                        rewards=rewards[prev_env_step.player_id].item(),
                        value=actor_step.value,
                    ),
                    ts=num_steps,
                )
            )

            num_steps += 1
            if dones >= 2:
                actor_step = _actor_step(worker_index, model, obs, verbose)
                timesteps[env_step.player_id].append(
                    TimeStep(
                        id=f"{worker_index}{env_step.player_id}",
                        env=env_step,
                        actor=ActorStep(
                            action=actor_step.action,
                            policy=actor_step.policy,
                            rewards=rewards[env_step.player_id].item(),
                            value=actor_step.value,
                        ),
                        ts=num_steps,
                    )
                )
                _reset(worker_index, learn_queue, eval_queue, num_battles, timesteps)
                await _drain(player_index, env)
                for player_index in range(2):
                    timesteps[player_index] = []
                break

        num_battles += 1


def run_environment(
    worker_index: int,
    model: nn.Module,
    learn_queue: mp.Queue,
    eval_queue: mp.Queue,
    verbose: bool = False,
    threshold: int = 1,
):
    return asyncio.run(
        _run_environment_async(
            worker_index, model, learn_queue, eval_queue, verbose, threshold
        )
    )
