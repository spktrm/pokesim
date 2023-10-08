import os

os.environ["OMP_NUM_THREADS"] = "1"

import time
import torch
import wandb
import random
import threading
import traceback

import torch.nn as nn
import multiprocessing as mp

from typing import List

from pokesim.data import MODEL_INPUT_KEYS
from pokesim.env import Environment
from pokesim.structs import (
    ActorStep,
    EnvStep,
    ModelOutput,
    TimeStep,
)

from pokesim.structs import Batch, Trajectory
from pokesim.learner import Learner


def run_environment_wrapper(worker_index: int, model: nn.Module, learn_queue: mp.Queue):
    try:
        run_environment(worker_index, model, learn_queue)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        traceback.print_exc()
        raise e


def run_environment(worker_index: int, model: nn.Module, learn_queue: mp.Queue):
    env = Environment(worker_index)

    model_output: ModelOutput

    while True:
        obs, player_index = env.reset()

        timesteps = []
        while True:
            model_input = {
                key: torch.from_numpy(obs[key][None, None, ...])
                for key in MODEL_INPUT_KEYS
            }
            with torch.no_grad():
                model_output = model(**model_input)

            possible_choices = list(range(12))
            policy = model_output.policy.cpu().flatten().numpy()
            action = random.choices(possible_choices, weights=policy.tolist())[0]

            next_obs, reward, done, player_index = env.step(action)

            env_step = EnvStep(
                game_id=worker_index,
                player_id=player_index,
                state=obs["raw"],
                rewards=reward,
                valid=True,
                legal=obs["legal"],
                history_mask=obs["history_mask"],
            )
            actor_step = ActorStep(policy=policy, action=action)
            timestep = TimeStep(id="", actor=actor_step, env=env_step)
            timesteps.append(timestep)

            obs = next_obs

            if done:
                trajectory = Trajectory.from_env_steps(timesteps)
                learn_queue.put(trajectory.serialize())
                del timesteps[:]

                break


class ReplayBuffer:
    def __init__(self, queue: mp.Queue, max_buffer_size: int = 512):
        self.queue = queue
        self.max_buffer_size = max_buffer_size

        self.buffer = []

    def sample(self, batch_size: int = 16, lock=threading.Lock()):
        with lock:
            batch = [self.queue.get() for _ in range(batch_size)]

        self.buffer += batch
        if len(self.buffer) > self.max_buffer_size:
            self.buffer = self.buffer[-self.max_buffer_size :]

        return Batch.from_trajectories(
            [
                Trajectory.deserialize(self.buffer[index])
                for index in random.sample(range(len(self.buffer)), batch_size)
            ]
        )


def learn(learner: Learner, batch: Batch, lock=threading.Lock()):
    with lock:
        alpha, update_target_net = learner._entropy_schedule(learner.learner_steps)
        logs = learner.update_parameters(batch, alpha, update_target_net)

        learner.learner_steps += 1

        logs["avg_length"] = batch.valid.sum(0).mean()
        logs["learner_steps"] = learner.learner_steps
        return logs


def learn_loop(learner: Learner, queue: mp.Queue):
    # progress = tqdm(desc="Learning")
    env_steps = 0

    replay_buffer = ReplayBuffer(queue)

    while True:
        batch = replay_buffer.sample(learner.config.batch_size)
        env_steps += batch.valid.sum()

        logs = learn(learner, batch)
        # logs["env_steps"] = env_steps

        wandb.log(logs)


def main():
    init = None
    learner = Learner(init)

    wandb.init(
        # set the wandb project where this run will be logged
        project="pokesim",
        # track hyperparameters and run metadata
        config=learner.get_config(),
    )

    processes: List[mp.Process] = []
    threads: List[threading.Thread] = []

    num_workers = 12

    learn_queue = mp.Queue(maxsize=max(36, learner.config.batch_size))

    for worker_index in range(num_workers):
        process = mp.Process(
            target=run_environment_wrapper,
            args=(worker_index, learner.params_actor, learn_queue),
        )
        process.start()
        processes.append(process)

    for _ in range(1):
        learn_thread = threading.Thread(
            target=learn_loop,
            args=(learner, learn_queue),
        )
        threads.append(learn_thread)
        learn_thread.start()

    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        return
    else:
        for thread in threads:
            thread.join()
    finally:
        for process in processes:
            process.join()


if __name__ == "__main__":
    mp.set_start_method("fork")
    main()
