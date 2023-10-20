from typing import Dict, List
import torch
import random

import numpy as np
import torch.nn as nn
import multiprocessing as mp


from pokesim.data import EVAL_WORKER_INDEX, MODEL_INPUT_KEYS, PREV_WORKER_INDEX
from pokesim.env import Environment
from pokesim.structs import (
    ActorStep,
    EnvStep,
    TimeStep,
)

from pokesim.structs import Trajectory


def run_environment(
    worker_index: int,
    model: nn.Module,
    model_prev: nn.Module,
    learn_queue: mp.Queue,
    eval_queue: mp.Queue,
):
    env = Environment(worker_index)

    num_battles = 0

    action_space = list(range(12))

    while True:
        obs, player_index = env.reset()
        reward = np.array([0, 0])

        env_step = EnvStep(
            game_id=worker_index,
            player_id=player_index,
            state=obs["raw"],
            rewards=reward,
            valid=True,
            legal=obs["legal"],
            history_mask=obs["history_mask"],
        )

        timesteps: Dict[int, List[TimeStep]] = {0: [], 1: []}
        with torch.no_grad():
            while True:
                prev_env_step = env_step

                model_input = {
                    key: torch.from_numpy(obs[key][None, None, ...])
                    for key in MODEL_INPUT_KEYS
                }

                if worker_index == PREV_WORKER_INDEX and player_index == 1:
                    actor_model = model_prev
                else:
                    actor_model = model

                pi, *_, value = actor_model(**model_input)
                pi = pi.cpu().numpy().flatten()
                value = value.cpu().numpy().flatten()
                action = random.choices(action_space, weights=pi)[0]

                obs, reward, done, player_index = env.step(action)

                env_step = EnvStep(
                    game_id=worker_index,
                    player_id=player_index,
                    state=obs["raw"],
                    rewards=reward[player_index],
                    valid=not done,
                    legal=obs["legal"],
                    history_mask=obs["history_mask"],
                )

                if worker_index < EVAL_WORKER_INDEX:
                    actor_step = ActorStep(
                        policy=pi, action=action, value=value, rewards=env_step.rewards
                    )
                    timestep = TimeStep(id="", actor=actor_step, env=prev_env_step)
                    timesteps[prev_env_step.player_id].append(timestep)

                if done:
                    if worker_index < EVAL_WORKER_INDEX:
                        for player_index in range(2):
                            timesteps[player_index][-1] = TimeStep(
                                id="",
                                actor=ActorStep(
                                    policy=timesteps[player_index][-1].actor.policy,
                                    action=timesteps[player_index][-1].actor.action,
                                    value=timesteps[player_index][-1].actor.value,
                                    rewards=reward[player_index],
                                ),
                                env=timesteps[player_index][-1].env,
                            )
                            trajectory = Trajectory.from_env_steps(
                                timesteps[player_index]
                            )
                            learn_queue.put(trajectory.serialize())

                    else:
                        eval_queue.put((num_battles, worker_index, reward[0]))

                    for player_index in range(2):
                        del timesteps[player_index][:]

                    num_battles += 1
                    break
