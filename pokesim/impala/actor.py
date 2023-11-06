import torch
import random

import numpy as np
import torch.nn as nn
import multiprocessing as mp


from pokesim.data import EVAL_WORKER_INDEX, MODEL_INPUT_KEYS, PREV_WORKER_INDEX
from pokesim.env import EnvironmentNoStack
from pokesim.structs import (
    ActorStep,
    EnvStep,
    ModelOutput,
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


_MODEL_INPUT_KEYS = MODEL_INPUT_KEYS.copy()
_MODEL_INPUT_KEYS.remove("history_mask")


def run_environment(
    worker_index: int,
    model: nn.Module,
    model_prev: nn.Module,
    learn_queue: mp.Queue,
    eval_queue: mp.Queue,
    verbose: bool = False,
):
    env = EnvironmentNoStack(worker_index)

    num_battles = 0
    model_output: ModelOutput
    action_space = list(range(12))

    with torch.no_grad():
        while True:
            n = 0

            obs, player_index = env.reset()

            env_step = EnvStep(
                game_id=worker_index,
                player_id=player_index,
                state=obs["raw"],
                rewards=np.zeros((1,)),
                valid=True,
                legal=obs["legal"],
            )

            timesteps = {
                0: [],
                1: [],
            }
            hidden_states = {
                0: model.get_hidden_state(1),
                1: model.get_hidden_state(1),
            }

            while True:
                prev_env_step: EnvStep = env_step

                model_input = {
                    key: torch.from_numpy(obs[key]) for key in _MODEL_INPUT_KEYS
                }

                if worker_index == PREV_WORKER_INDEX and player_index == 1:
                    actor_model = model_prev
                else:
                    actor_model = model

                model_output = actor_model(
                    **model_input, hidden_state=hidden_states[player_index]
                )

                pi = model_output.policy
                value = model_output.value
                hidden_states[player_index] = model_output.hidden_state

                pi = pi.cpu().numpy().flatten()
                value = value.cpu().numpy().flatten()
                action = random.choices(action_space, weights=pi)[0]

                if verbose:
                    handle_verbose(n, pi, action, value)

                obs, reward, done, player_index = env.step(action)

                env_step = EnvStep(
                    game_id=worker_index,
                    player_id=player_index,
                    state=obs["raw"],
                    rewards=reward,
                    valid=True,
                    legal=obs["legal"],
                )

                actor_step = ActorStep(
                    policy=pi,
                    action=action,
                    rewards=prev_env_step.rewards,
                    value=value,
                )

                timestep = TimeStep(id="", actor=actor_step, env=prev_env_step)
                timesteps[prev_env_step.player_id].append(timestep)

                if done:
                    if worker_index < EVAL_WORKER_INDEX:
                        for player_index in range(2):
                            if env_step.player_id == player_index:
                                timestep = TimeStep(
                                    id="",
                                    actor=ActorStep(
                                        policy=pi,
                                        action=action,
                                        rewards=env_step.rewards,
                                        value=value,
                                    ),
                                    env=env_step,
                                )
                                timesteps[env_step.player_id].append(timestep)
                            trajectory = Trajectory.from_env_steps(
                                timesteps[player_index], fix_rewards=False
                            )
                            learn_queue.put(trajectory.serialize())

                    else:
                        if env_step.player_id == 0:
                            timestep = TimeStep(
                                id="",
                                actor=ActorStep(
                                    policy=pi,
                                    action=action,
                                    rewards=env_step.rewards,
                                    value=value,
                                ),
                                env=env_step,
                            )
                            timesteps[env_step.player_id].append(timestep)

                        final_reward = timesteps[0][-1].actor.rewards.item()
                        eval_queue.put((num_battles, worker_index, final_reward))

                    num_battles += 1
                    break

                n += 1
