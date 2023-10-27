import os


os.environ["OMP_NUM_THREADS"] = "1"

import time
import torch
import wandb
import threading
import traceback

import multiprocessing as mp

from tqdm import tqdm
from typing import List

from pokesim.data import EVAL_MAPPING, NUM_WORKERS
from pokesim.structs import Batch, Trajectory

from pokesim.rnad.learner import Learner
from pokesim.rnad.actor import run_environment


def run_environment_wrapper(*args, **kwargs):
    try:
        run_environment(*args, **kwargs)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        traceback.print_exc()
        raise e


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
            tuple(
                [
                    Trajectory.deserialize(queue.get())
                    for _ in range(learner.config.batch_size)
                ]
            )
        )
        env_steps += batch.valid.sum()

        logs = learn(learner, batch)
        # logs["env_steps"] = env_steps

        wandb.log(logs)
        progress.update(1)


def main(debug):
    # init = torch.load("ckpts/002432.pt", map_location="cpu")
    # init = init["params"]
    init = None
    learner = Learner(init=init, debug=debug)
    # learner = Learner.from_fpath("ckpts/113083.pt")

    if not debug:
        config = learner.get_config()
        wandb.init(
            # set the wandb project where this run will be logged
            project="pokesim",
            # track hyperparameters and run metadata
            config=config,
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
            args=(
                worker_index,
                learner.params_actor,
                learner.params_actor_prev,
                learn_queue,
                eval_queue,
            ),
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

        if not os.path.exists("ckpts"):
            os.mkdir("ckpts")

        while True:
            time.sleep(1)

            if (time.time() - prev_time) >= 60 * 60:
                learner.save(
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
