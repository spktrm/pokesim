import os

os.environ["OMP_NUM_THREADS"] = "1"

import time
import wandb
import uvloop
import random
import asyncio

import torch

torch.set_float32_matmul_precision("high")

import torch.nn as nn

import threading
import multiprocessing as mp

from typing import List
from tqdm import tqdm

from pokesim.types import OpponentPolicy
from pokesim.inference import Inference
from pokesim.manager import Manager
from pokesim.actor import (
    SelfplayActor,
    DefaultEvalActor,
    RandomEvalActor,
    MaxdmgEvalActor,
)
from pokesim.structs import Batch, Trajectory
from pokesim.learner import Learner

_DEBUG = False
_NUM_ACTORS = 1 if _DEBUG else 14
print(f"Running on `{_NUM_ACTORS} Workers`")


async def _run_worker(
    worker_index: int,
    model: nn.Module,
    progress_queue: mp.Queue,
    queue: mp.Queue,
    selfplay: bool = True,
    opponent_pi: OpponentPolicy = None,
):
    inference = Inference(model, "cpu", batch_size=2)
    command = f"node ./sim.js {worker_index}"

    process = await asyncio.create_subprocess_shell(
        command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    if selfplay:
        opponent = SelfplayActor(inference)

    if opponent_pi is not None:
        if opponent_pi == "default":
            opponent = DefaultEvalActor(opponent_pi, queue)
        elif opponent_pi == "maxdmg":
            opponent = MaxdmgEvalActor(opponent_pi, queue)
        elif opponent_pi == "random":
            opponent = RandomEvalActor(opponent_pi, queue)
        else:
            raise ValueError

    manager = Manager(
        worker_index=worker_index,
        process=process,
        actor1=SelfplayActor(inference),
        actor2=opponent,
    )

    if selfplay:
        learn_queue = queue
    else:
        learn_queue = None

    await manager.run(progress_queue, learn_queue)


def run_worker(
    worker_index: int,
    model: nn.Module,
    progress_queue: mp.Queue,
    learn_queue: mp.Queue,
    selfplay: bool = True,
    opponent_pi: OpponentPolicy = None,
):
    asyncio.run(
        _run_worker(
            worker_index,
            model,
            progress_queue,
            learn_queue,
            selfplay,
            opponent_pi,
        )
    )


def read_prog(progress_queue: mp.Queue):
    progress1 = tqdm(desc="games")
    progress2 = tqdm(desc="steps", leave=False)
    while True:
        l = progress_queue.get()
        progress1.update(1)
        progress2.update(l)


def read_eval(eval_queue: mp.Queue):
    while True:
        n, o, r = eval_queue.get()
        if not _DEBUG:
            wandb.log({f"{o}_n": n, f"{o}_r": r})


def learn(learner: Learner, batch: Batch, lock=threading.Lock()):
    with lock:
        alpha, update_target_net = learner._entropy_schedule(learner.learner_steps)
        logs = learner.update_parameters(batch, alpha, update_target_net)

        learner.learner_steps += 1

        logs["avg_length"] = batch.valid.sum(0).mean()
        logs["learner_steps"] = learner.learner_steps
        return logs


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


def learn_loop(learner: Learner, queue: mp.Queue):
    # progress = tqdm(desc="Learning")
    env_steps = 0

    replay_buffer = ReplayBuffer(queue)

    while True:
        batch = replay_buffer.sample(learner.config.batch_size)
        env_steps += batch.valid.sum()

        logs = learn(learner, batch)
        # logs["env_steps"] = env_steps

        if not _DEBUG:
            wandb.log(logs)


def main():
    init = None
    init = torch.load("ckpts/ckpt-125335.pt")
    learner = Learner(init)

    if not _DEBUG:
        wandb.init(
            # set the wandb project where this run will be logged
            project="meloettav2",
            # track hyperparameters and run metadata
            config=learner.get_config(),
        )

    progress_queue = mp.Queue()
    eval_queue = mp.Queue()
    learn_queue = mp.Queue(maxsize=max(36, learner.config.batch_size))

    progress_thread = threading.Thread(target=read_prog, args=(progress_queue,))
    progress_thread.start()

    eval_thread = threading.Thread(target=read_eval, args=(eval_queue,))
    eval_thread.start()

    learn_threads: List[threading.Thread] = []
    for _ in range(1):
        learn_thread = threading.Thread(
            target=learn_loop,
            args=(learner, learn_queue),
        )
        learn_threads.append(learn_thread)
        learn_thread.start()

    uvloop.install()

    procs: List[mp.Process] = []
    if _DEBUG:
        opponent_policies = {}
    else:
        opponent_policies = {
            "default",
            "random",
            # "maxdmg",
        }

    procs += [
        mp.Process(
            target=run_worker,
            args=(
                len(procs) + 1,
                learner.params_actor,
                progress_queue,
                eval_queue,
                False,
                opponent_policy,
            ),
        )
        for opponent_policy in opponent_policies
    ]
    procs += [
        mp.Process(
            target=run_worker,
            args=(worker_index, learner.params_actor, progress_queue, learn_queue),
        )
        for worker_index in range(_NUM_ACTORS - len(procs))
    ]

    for proc in procs:
        proc.start()

    def checkpoint():
        torch.save(
            learner.params.state_dict(), f"ckpts/ckpt-{learner.learner_steps}.pt"
        )

    timer = time.time
    try:
        last_checkpoint_time = timer()
        while True:
            time.sleep(5)

            if timer() - last_checkpoint_time > 5 * 60:
                checkpoint()
                last_checkpoint_time = timer()

    except KeyboardInterrupt:
        return
    else:
        for thread in learn_threads:
            thread.join()
        progress_thread.join()
        eval_thread.join()
        for proc in procs:
            proc.join()


if __name__ == "__main__":
    main()
