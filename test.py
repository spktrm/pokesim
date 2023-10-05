import os

from git import Sequence

from pokesim.constants import _NUM_HISTORY

os.environ["OMP_NUM_THREADS"] = "1"

import torch
import wandb
import random
import socket
import threading

import numpy as np
import torch.nn as nn
import multiprocessing as mp

from tqdm import tqdm
from typing import Tuple, List
from pokesim.data import SOCKET_PATH, ENCODING, NUM_WORKERS
from pokesim.structs import Observation, State

from pokesim.manager import Manager
from pokesim.actor import (
    SelfplayActor,
    DefaultEvalActor,
    RandomEvalActor,
    MaxdmgEvalActor,
)
from pokesim.structs import Batch, Trajectory
from pokesim.learner import Learner

torch.set_float32_matmul_precision("high")


def read(sock: socket.socket) -> Tuple[int, bytes]:
    num_states = int.from_bytes(sock.recv(1), "big")
    data = b""
    while len(data) < num_states * 518:
        remaining = (518 * num_states) - len(data)
        data += sock.recv(remaining)
    return num_states, data


class Environment:
    def __init__(self, socket_address: str = SOCKET_PATH):
        self.socket_address = socket_address
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        print(f"Connecting to {socket_address}")
        self.sock.connect(socket_address)

    def read_stdout(self):
        num_states, out = read(self.sock)
        arr = np.frombuffer(out, dtype=np.int8)
        return arr.reshape(-1, 518)

    def reset(self) -> Observation:
        arr = self.read_stdout()
        return Observation(arr)

    def step(self, action: str) -> Observation:
        self.sock.sendall(action.encode(ENCODING))
        return self.reset()


class Model:
    def __init__(self):
        self.w1 = np.random.random((128, 128))
        self.w2 = np.random.random((128, 128))
        self.w3 = np.random.random((128, 128))
        self.w4 = np.random.random((128, 10))

    def forward(self, state: np.ndarray, legal: np.ndarray):
        x = state @ self.w1
        x = x @ self.w2
        x = x @ self.w3
        logit = x @ self.w4

        logit = np.where(legal, logit, 0)
        exp_logit = logit
        exp_logit_sum = exp_logit.sum(axis=-1, keepdims=True)
        prob = exp_logit / exp_logit_sum
        return prob


def stacknpad(array_stack: Sequence[np.ndarray], num_padding: int):
    stack = np.stack(array_stack)
    return np.concatenate(
        (stack, np.tile(stack[-1, None], (num_padding - stack.shape[0], 1)))
    )


def run_environment(model: nn.Module, learn_queue: mp.Queue):
    env = Environment()

    buffer = {i: [] for i in range(NUM_WORKERS)}
    dones = {i: 0 for i in range(NUM_WORKERS)}
    hist = {i: {0: [], 1: []} for i in range(NUM_WORKERS)}

    obs = env.reset()
    while True:
        states = obs.get_state()
        legal = obs.get_legal_moves()
        worker_indices = obs.get_worker_index()
        player_indices = obs.get_player_index()

        batch = []

        for state_index, (done, worker_index, player_index) in enumerate(
            zip(
                obs.get_done(),
                worker_indices,
                player_indices,
            )
        ):
            state = states[state_index]
            observation = obs.obs[state_index]
            hist[worker_index][player_index].append(observation)

            state_stack = stacknpad(
                hist[worker_index][player_index][-_NUM_HISTORY:], _NUM_HISTORY
            )

            num_states = len(hist[worker_index][player_index][-_NUM_HISTORY:])
            item = (state_stack, num_states)

            buffer[worker_index].append(item)
            batch.append(item)

            dones[worker_index] += done

            if dones[worker_index] >= 2:
                trajectory, stack_lengths = list(zip(*buffer[worker_index]))

                trajectory = np.stack(trajectory)
                stack_lengths = np.concatenate(stack_lengths)

                buffer[worker_index] = []
                hist[worker_index] = {0: [], 1: []}
                dones[worker_index] = 0

        batch, stack_lengths = list(zip(*batch))
        state = State(np.stack(batch)).dense()

        probs = model(**state, legal=legal)

        actions = []
        for worker_index, player_index, prob in zip(
            worker_indices, player_indices, probs.tolist()
        ):
            action = random.choices(range(10), weights=prob)
            actions.append(f"{worker_index}|{player_index}|{action[0]}")

        actions = "\n".join(actions)
        obs = env.step(actions)


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

    processes = []
    learn_queue = mp.Queue(maxsize=max(36, learner.config.batch_size))

    for i in range(1):
        proc = mp.Process(
            target=run_environment,
            args=(
                learner.params_actor,
                learn_queue,
            ),
        )
        proc.start()
        processes.append(proc)

    learn_threads: List[threading.Thread] = []
    for _ in range(1):
        learn_thread = threading.Thread(
            target=learn_loop,
            args=(learner, learn_queue),
        )
        learn_threads.append(learn_thread)
        learn_thread.start()

    for proc in processes:
        proc.join()


if __name__ == "__main__":
    main()
