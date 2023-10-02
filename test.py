import yaml
import random
import socket
import numpy as np
import multiprocessing as mp

from typing import NamedTuple

NUM_WORKERS = 20

with open("./config.yml", "r") as f:
    CONFIG = yaml.safe_load(f)

SOCKET_PATH = CONFIG["socket_path"]
ENCODING = CONFIG["encoding"]


def read(sock: socket.socket) -> bytes:
    num_states = int.from_bytes(sock.recv(1), "big")
    data = sock.recv(518 * num_states)
    return data


class Step(NamedTuple):
    states: np.ndarray
    legal_actions: np.ndarray
    dones: np.ndarray
    rewards: np.ndarray
    worker_indices: np.ndarray
    player_indices: np.ndarray


class Environment:
    def __init__(self, socket_address: str = SOCKET_PATH):
        self.socket_address = socket_address
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        print(f"Connecting to {socket_address}")
        self.sock.connect(socket_address)

    def read_stdout(self):
        out = read(self.sock)
        arr = np.frombuffer(out, dtype=np.int8)
        return arr.reshape(-1, 518)

    def reset(self) -> Step:
        arr = self.read_stdout()
        states = arr[..., 4:-10]
        legal_actions = arr[..., -10:]
        worker_indices = arr[..., 0]
        player_indices = arr[..., 1]
        dones = arr[..., 2]
        rewards = arr[..., 3]
        return Step(
            states, legal_actions, dones, rewards, worker_indices, player_indices
        )

    def step(self, action: str) -> Step:
        self.sock.sendall(action.encode(ENCODING))
        return self.reset()


def run_environment():
    env = Environment()

    buffer = {i: [] for i in range(NUM_WORKERS)}
    dones = {i: 0 for i in range(NUM_WORKERS)}

    obs = env.reset()
    while True:
        actions = []

        for state, legal, done, reward, worker_index, player_index in zip(*obs):
            buffer[worker_index].append(state)
            dones[worker_index] += done

            if dones[worker_index] >= 2:
                trajectory = np.stack(buffer[worker_index])

                buffer[worker_index] = []
                dones[worker_index] = 0

            action = random.choices(range(10), weights=legal.tolist())
            actions.append(f"{worker_index}|{player_index}|{action[0]}")

        actions = "\n".join(actions)
        obs = env.step(actions)


def main():
    processes = []

    for i in range(1):
        proc = mp.Process(target=run_environment)
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()


if __name__ == "__main__":
    main()
