import socket
import numpy as np

from typing import Sequence, Tuple, Dict

from pokesim.data import SOCKET_PATH, ENCODING, NUM_HISTORY
from pokesim.structs import Observation, State


def read(sock: socket.socket, state_size: int = 526) -> bytes:
    data = b""
    while len(data) < state_size:
        remaining = state_size - len(data)
        data += sock.recv(remaining)
    return data


def stacknpad(array_stack: Sequence[np.ndarray], num_padding: int):
    stack = np.stack(array_stack)
    return np.concatenate(
        (stack, np.tile(stack[-1, None], (num_padding - stack.shape[0], 1)))
    )


class Environment:
    def __init__(self, worker_index: int, socket_address: str = SOCKET_PATH):
        self.socket_address = socket_address
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.worker_index = worker_index
        print(f"Worker {worker_index} Connecting to {socket_address}")
        self.sock.connect(socket_address)
        print(f"Worker {worker_index} Connected successfully!")
        self.reset_env_vars()

    def reset_env_vars(self):
        self.dones = np.array([0, 0], dtype=bool)
        self.history = {0: [], 1: []}
        self.policy_select = 0
        self.current_player = 0
        self.reward = np.array([0, 0])

    def is_done(self):
        return self.dones.all()

    def read_stdout(self):
        out = read(self.sock)
        return np.frombuffer(out, dtype=np.int8)

    def recvenv(self) -> Observation:
        arr = self.read_stdout()
        self.observation = Observation(arr)

    def reset(self):
        self.reset_env_vars()
        self.recvenv()
        state, *_, player_index = self.process_state()
        return state, player_index

    def is_current_player_done(self):
        return self.dones[self.current_player] == 1

    def step(self, action_index: int):
        if action_index > 1:
            if not self.is_current_player_done():
                action = f"{self.current_player}|{action_index-2}"
                self.sock.sendall(action.encode(ENCODING))
            self.policy_select = 0
            self.recvenv()
        else:
            self.policy_select = action_index + 1

        return self.process_state()

    def process_state(self) -> Tuple[Dict[str, np.ndarray], int, bool]:
        if self.policy_select == 0:
            player_index = self.observation.get_player_index()
            self.current_player = player_index.item()
            done = self.observation.get_done()
            self.dones[self.current_player] |= bool(done)
            state = self.observation.get_state()
            self.reward[self.current_player] = self.observation.get_reward()
            self.history[self.current_player].append(state)
            num_states = len(self.history[self.current_player][-NUM_HISTORY:])
            state_stack = State(
                stacknpad(self.history[self.current_player][-NUM_HISTORY:], NUM_HISTORY)
            )
            history_mask = num_states >= np.arange(1, NUM_HISTORY + 1)
            state = state_stack.dense()
            self.prev_state = dict(
                **state,
                history_mask=history_mask,
            )
        legal_moves = self.observation.get_legal_moves(self.policy_select)
        self.prev_state["legal"] = legal_moves
        is_done = self.is_done()
        return (self.prev_state, is_done * self.reward, is_done, self.current_player)
