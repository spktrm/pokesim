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
from pokesim.utils import finetune


def handle_verbose(n: int, pi: np.ndarray, action: np.ndarray, value: np.ndarray):
    nice_probs = [f"{p:.2f}" for p in pi]
    action_type = " ".join(nice_probs[:2])
    move = " ".join(nice_probs[2:6])
    switch = " ".join(nice_probs[6:])
    v = f"{value.item():.3f}"

    text = "\n".join(
        [
            str(n),
            action_type,
            move,
            switch,
            f"{action}",
            v,
        ]
    )
    print(text + "\n")


def run_environment(
    worker_index: int,
    model: nn.Module,
    model_prev: nn.Module,
    learn_queue: mp.Queue,
    eval_queue: mp.Queue,
    verbose: bool = False,
):
    env = Environment(worker_index)

    num_battles = 0

    action_space = list(range(12))

    with torch.no_grad():
        while True:
            n = 0
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

            timesteps = []
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
                if worker_index >= EVAL_WORKER_INDEX:
                    pi = finetune(pi, model_input["legal"])
                pi = pi.cpu().numpy().flatten()
                value = value.cpu().numpy().flatten()
                action = random.choices(action_space, weights=pi)[0]

                if verbose:
                    handle_verbose(n, pi, action, value)

                obs, reward, done, player_index, player_is_done = env.step(action)

                env_step = EnvStep(
                    game_id=worker_index,
                    player_id=player_index,
                    state=obs["raw"],
                    rewards=reward.copy(),
                    valid=not player_is_done,
                    legal=obs["legal"],
                    history_mask=obs["history_mask"],
                )

                if worker_index < EVAL_WORKER_INDEX:
                    actor_step = ActorStep(
                        policy=pi, action=action, rewards=env_step.rewards, value=value
                    )
                    timestep = TimeStep(id="", actor=actor_step, env=prev_env_step)
                    timesteps.append(timestep)

                if done:
                    if worker_index < EVAL_WORKER_INDEX:
                        trajectory = Trajectory.from_env_steps(timesteps)
                        learn_queue.put(trajectory.serialize())

                    else:
                        final_reward = env_step.rewards[player_index]
                        if player_index == 1:
                            final_reward = -final_reward
                        eval_queue.put((num_battles, worker_index, final_reward))

                    del timesteps[:]

                    num_battles += 1
                    break

                n += 1
