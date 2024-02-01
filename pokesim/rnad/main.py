import os

os.environ["OMP_NUM_THREADS"] = "1"

import torch

import time
import wandb
import threading
import traceback

import torch.multiprocessing as mp
from multiprocessing.context import ForkContext

from tqdm import tqdm
from typing import Any, Dict, List

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


class WandbLogger:
    def __init__(self, accum_steps: int):
        self.accum_steps = accum_steps
        self.history = []
        self.count = 0

    def aggregate(self):
        agg_log = {}
        for log in self.history:
            for key, value in log.items():
                if key not in agg_log:
                    agg_log[key] = 0
                agg_log[key] += value
        for key, value in agg_log.items():
            agg_log[key] = value / self.accum_steps
        agg_log["learner_steps"] = self.history[-1]["learner_steps"]
        self.history = []
        return agg_log

    def __call__(self, logs: Dict[str, Any]):
        self.history.append(logs)
        if self.count % self.accum_steps == 0:
            agg_logs = self.aggregate()
            wandb.log(agg_logs)
        self.count += 1


def learn_loop(
    learner: Learner,
    queue: mp.Queue,
    condition: mp.Condition,
    debug: bool = False,
):
    progress = tqdm(desc="Learning")
    env_steps = 0
    wandb_logger = WandbLogger(learner.config.accum_steps)

    while True:
        batch = []

        for _ in range(learner.config.batch_size):
            trajectory = queue.get()
            batch.append(Trajectory.deserialize(trajectory))

        batch = Batch.from_trajectories(batch)
        env_steps += batch.valid.sum()

        logs = learn(learner, batch)
        # logs["env_steps"] = env_steps

        # with condition:
        #     condition.notify_all()

        if not debug:
            wandb_logger(logs)
        progress.update(1)


def main(ctx: ForkContext = ForkContext(), debug: bool = False):
    # fpath = get_most_recent_file("ckpts")
    # print(fpath)
    # init = torch.load(fpath, map_location="cpu")
    # init = init["params"]

    init = None
    learner = Learner(init=init, debug=debug, trace_nets=False)  # not debug)

    # learner = Learner.from_fpath(fpath, trace_nets=False)

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

    processes: List[ctx.Process] = []
    threads: List[threading.Thread] = []

    learn_queue = ctx.Queue(maxsize=2 * learner.config.batch_size)
    eval_queue = ctx.Queue()

    if not debug:
        eval_thread = threading.Thread(target=read_eval, args=(eval_queue,))
        threads.append(eval_thread)
        eval_thread.start()

    batch_condition = ctx.Condition()
    for worker_index in range(num_workers):
        process = ctx.Process(
            target=run_environment_wrapper,
            args=(
                worker_index,
                learner.params_actor,
                None,
                learn_queue,
                eval_queue,
                batch_condition,
            ),
        )
        process.start()
        processes.append(process)

    for _ in range(1):
        learn_thread = threading.Thread(
            target=learn_loop,
            args=(learner, learn_queue, batch_condition, debug),
        )
        learn_thread.start()
        threads.append(learn_thread)

    try:
        prev_time = time.time()

        if not os.path.exists("ckpts"):
            os.mkdir("ckpts")

        while True:
            time.sleep(1)

            if (time.time() - prev_time) >= 30 * 60:
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
    ctx = mp.get_context("fork")
    main(ctx, False)
