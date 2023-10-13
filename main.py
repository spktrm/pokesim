import os

os.environ["OMP_NUM_THREADS"] = "1"

import time
import torch
import wandb
import random
import threading
import traceback

import numpy as np
import torch.nn as nn
import multiprocessing as mp

from tqdm import tqdm
from typing import List

from pokesim.data import (
    EVAL_MAPPING,
    EVAL_WORKER_INDEX,
    NUM_WORKERS,
    MODEL_INPUT_KEYS,
)
from pokesim.env import Environment
from pokesim.structs import (
    ActorStep,
    EnvStep,
    TimeStep,
)

from pokesim.structs import Batch, Trajectory
from pokesim.learners.rnad_learner import Learner


def run_environment_wrapper(
    worker_index: int,
    model: nn.Module,
    learn_queue: mp.Queue,
    eval_queue: mp.Queue,
):
    try:
        run_environment(worker_index, model, learn_queue, eval_queue)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        traceback.print_exc()
        raise e


def run_environment(
    worker_index: int,
    model: nn.Module,
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

        timesteps = []
        while True:
            prev_env_step = env_step

            model_input = {
                key: torch.from_numpy(obs[key][None, None, ...])
                for key in MODEL_INPUT_KEYS
            }
            with torch.no_grad():
                pi, *_ = model(**model_input)

            pi = pi.cpu().numpy().flatten()
            action = random.choices(action_space, weights=pi.tolist())[0]

            obs, reward, done, player_index = env.step(action)

            env_step = EnvStep(
                game_id=worker_index,
                player_id=player_index,
                state=obs["raw"],
                rewards=reward,
                valid=not done,
                legal=obs["legal"],
                history_mask=obs["history_mask"],
            )

            if worker_index < EVAL_WORKER_INDEX:
                actor_step = ActorStep(
                    policy=pi, action=action, rewards=env_step.rewards
                )
                timestep = TimeStep(id="", actor=actor_step, env=prev_env_step)
                timesteps.append(timestep)

            if done:
                if worker_index < EVAL_WORKER_INDEX:
                    trajectory = Trajectory.from_env_steps(timesteps)
                    learn_queue.put(trajectory.serialize())

                else:
                    eval_queue.put((num_battles, worker_index, env_step.rewards[0]))

                del timesteps[:]

                num_battles += 1
                break


def read_eval(eval_queue: mp.Queue):
    while True:
        n, o, r = eval_queue.get()
        pi = EVAL_MAPPING[o]
        wandb.log({f"{pi}_n": n, f"{pi}_r": r})


def learn(learner: Learner, batch: Batch, lock=threading.Lock()):
    with lock:
        alpha, update_target_net = learner._entropy_schedule(learner.learner_steps)
        logs = learner.update_parameters(batch, alpha, update_target_net)

        learner.learner_steps += 1

        logs["avg_length"] = batch.valid.sum(0).mean()
        logs["learner_steps"] = learner.learner_steps
        return logs


def learn_loop(learner: Learner, queue: mp.Queue):
    progress = tqdm(desc="Learning")
    env_steps = 0

    while True:
        batch = Batch.from_trajectories(
            [
                Trajectory.deserialize(queue.get())
                for _ in range(learner.config.batch_size)
            ]
        )
        env_steps += batch.valid.sum()

        logs = learn(learner, batch)
        # logs["env_steps"] = env_steps

        wandb.log(logs)
        progress.update(1)


def main(debug):
    init = None
    # init = torch.load("ckpts/027632.pt")
    learner = Learner(init)

    if not debug:
        wandb.init(
            # set the wandb project where this run will be logged
            project="pokesim",
            # track hyperparameters and run metadata
            config=learner.get_config(),
        )
        num_workers = NUM_WORKERS
    else:
        num_workers = 1

    processes: List[mp.Process] = []
    threads: List[threading.Thread] = []

    learn_queue = mp.Queue(maxsize=learner.config.batch_size)
    eval_queue = mp.Queue()

    if not debug:
        eval_thread = threading.Thread(target=read_eval, args=(eval_queue,))
        threads.append(eval_thread)
        eval_thread.start()

    for worker_index in range(num_workers):
        process = mp.Process(
            target=run_environment_wrapper,
            args=(worker_index, learner.params_actor, learn_queue, eval_queue),
        )
        process.start()
        processes.append(process)

    for _ in range(1):
        learn_thread = threading.Thread(
            target=learn_loop,
            args=(learner, learn_queue),
        )
        learn_thread.start()
        threads.append(learn_thread)

    try:
        prev_time = time.time()
        while True:
            time.sleep(1)

            if (time.time() - prev_time) >= 5 * 60:
                torch.save(
                    learner.params_actor.state_dict(),
                    f"ckpts/{learner.learner_steps:06}.pt",
                )
                prev_time = time.time()

    except KeyboardInterrupt:
        return
    else:
        if threads:
            for thread in threads:
                thread.join()
    finally:
        for process in processes:
            process.join()


if __name__ == "__main__":
    mp.set_start_method("fork")
    main(False)
