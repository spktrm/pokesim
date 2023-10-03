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


TURN_OFFSET = 0
TURN_SIZE = 1

TEAM_OFFSET = TURN_OFFSET + TURN_SIZE
TEAM_SIZE = 3 * 6 * 22

SIDE_CONDITION_OFFSET = TEAM_OFFSET + TEAM_SIZE
SIDE_CONDITION_SIZE = 2 * 15

VOLATILE_STATUS_OFFSET = SIDE_CONDITION_OFFSET + SIDE_CONDITION_SIZE
VOLATILE_STATUS_SIZE = 2 * 20

BOOSTS_OFFSET = VOLATILE_STATUS_OFFSET + VOLATILE_STATUS_SIZE
BOOSTS_SIZE = 2 * 7

FIELD_OFFSET = BOOSTS_OFFSET + BOOSTS_SIZE
FIELD_SIZE = 9 + 6

HISTORY_OFFSET = FIELD_OFFSET + FIELD_SIZE


class State(NamedTuple):
    raw: np.ndarray

    def dense(self):
        batch_size = self.raw.shape[0]
        turn = self.raw[..., 0].reshape(batch_size, TURN_SIZE)
        teams = np.frombuffer(
            self.raw[..., TEAM_OFFSET:SIDE_CONDITION_OFFSET].tobytes(), dtype=np.int16
        ).reshape(batch_size, 3, 6, -1)
        side_conditions = self.raw[
            ..., SIDE_CONDITION_OFFSET:VOLATILE_STATUS_OFFSET
        ].reshape(batch_size, 2, -1)
        volatile_status = self.raw[..., VOLATILE_STATUS_OFFSET:BOOSTS_OFFSET].reshape(
            batch_size, 2, -1
        )
        boosts = self.raw[..., BOOSTS_OFFSET:FIELD_OFFSET].reshape(batch_size, 2, -1)
        field = self.raw[..., FIELD_OFFSET:HISTORY_OFFSET].reshape(
            batch_size, FIELD_SIZE
        )
        history = self.raw[..., HISTORY_OFFSET:].reshape(batch_size, -1, 2)
        return (
            self.raw,
            turn,
            teams,
            side_conditions,
            volatile_status,
            boosts,
            field,
            history,
        )


class Observation(NamedTuple):
    obs: np.ndarray

    def get_state(self):
        return self.obs[..., 4:-10]

    def get_legal_moves(self):
        return self.obs[..., -10:]

    def get_worker_index(self):
        return self.obs[..., 0]

    def get_player_index(self):
        return self.obs[..., 1]

    def get_done(self):
        return self.obs[..., 2]

    def get_reward(self):
        return self.obs[..., 3]


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


def run_environment():
    env = Environment()
    model = Model()

    buffer = {i: [] for i in range(NUM_WORKERS)}
    dones = {i: 0 for i in range(NUM_WORKERS)}

    obs = env.reset()
    while True:
        states = obs.get_state()
        legal = obs.get_legal_moves()
        worker_indices = obs.get_worker_index()
        player_indices = obs.get_player_index()

        for state_index, (done, worker_index, player_index) in enumerate(
            zip(
                obs.get_done(),
                worker_indices,
                player_indices,
            )
        ):
            buffer[worker_index].append(states[state_index])
            dones[worker_index] += done

            if dones[worker_index] >= 2:
                trajectory = np.stack(buffer[worker_index])

                buffer[worker_index] = []
                dones[worker_index] = 0

        probs = model.forward(np.ones_like(states[..., :128]), legal)
        actions = []

        for worker_index, player_index, prob in zip(
            worker_indices, player_indices, probs.tolist()
        ):
            action = random.choices(range(10), weights=prob)
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
