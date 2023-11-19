import asyncio
import socket
import numpy as np

from typing import Callable, Sequence, Tuple, Dict

from pokesim.data import SOCKET_PATH, ENCODING, NUM_HISTORY
from pokesim.structs import Observation, State

STATE_SIZE = 542


def read(sock: socket.socket, state_size: int = STATE_SIZE) -> bytes:
    data = b""
    while len(data) < state_size:
        remaining = state_size - len(data)
        data += sock.recv(remaining)
    return data


async def read_async(sock: asyncio.StreamReader, state_size: int = STATE_SIZE) -> bytes:
    data = b""
    while len(data) < state_size:
        remaining = state_size - len(data)
        data += await sock.read(remaining)
    return data


def stacknpad(array_stack: Sequence[np.ndarray], num_padding: int):
    stack = np.stack(array_stack)
    return np.concatenate(
        (stack, np.tile(stack[-1, None], (num_padding - stack.shape[0], 1)))
    )


class Environment:
    def __init__(
        self,
        worker_index: int,
        socket_address: str = SOCKET_PATH,
        verbose: bool = False,
    ):
        self.socket_address = socket_address
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.worker_index = worker_index
        if verbose:
            print(f"Worker {worker_index} Connecting to {socket_address}")
        self.sock.connect(socket_address)
        if verbose:
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
                action = f"{self.current_player}|{action_index-2}\n"
                self.sock.sendall(action.encode(ENCODING))
            self.policy_select = 0
            self.recvenv()
        else:
            if not self.is_current_player_done():
                self.policy_select = action_index + 1

        state, reward, done, player_index = self.process_state()
        is_player_done = self.is_current_player_done()
        if is_player_done and not self.is_done():
            self.policy_select = 0
            self.recvenv()
        return state, reward, done, player_index, is_player_done

    def process_state(self) -> Tuple[Dict[str, np.ndarray], int, bool]:
        reward = np.zeros(2)
        if self.policy_select == 0:
            player_index = self.observation.get_player_index()
            self.current_player = player_index.item()
            done = self.observation.get_done()
            self.dones[self.current_player] |= bool(done)
            state = self.observation.get_state()
            reward[self.current_player] = self.observation.get_reward()
            self.history[self.current_player].append(state)
            self.history[self.current_player] = self.history[self.current_player][
                -NUM_HISTORY:
            ]
            num_states = len(self.history[self.current_player][-NUM_HISTORY:])
            state_stack = State(
                stacknpad(
                    self.history[self.current_player][-NUM_HISTORY:], NUM_HISTORY
                ).copy()
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
        return (
            self.prev_state,
            reward.copy(),  # * is_done,
            is_done,
            self.current_player,
        )


class EnvironmentNoStack:
    _NUM_HISTORY = 2

    worker_index: int
    act_fn: Callable
    reset_fn: Callable
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter

    @classmethod
    async def create(cls, worker_index: int, act_fn: Callable, reset_fn: Callable):
        reader, writer = await asyncio.open_unix_connection(SOCKET_PATH)
        self = cls()
        self.worker_index = worker_index
        self.act_fn = act_fn
        self.reset_fn = reset_fn
        self.reader = reader
        self.writer = writer
        self.reset_env_vars()
        return self

    async def run(self):
        while True:
            self.reset_fn()
            obs, player_index = await self.reset()
            reward = np.zeros((1,))
            valid = True
            done = False

            while True:
                action = self.act_fn(obs, reward, valid, player_index)
                obs, reward, done, player_index = await self.step(action)
                valid = not bool(reward)
                if done:
                    self.act_fn(obs, reward, valid, player_index)
                    break

    def reset_env_vars(self):
        self.dones = np.array([0, 0], dtype=bool)
        self.history = {0: [], 1: []}
        self.policy_select = 0
        self.current_player = 0
        self.reward = np.zeros(1)

    def is_done(self):
        return self.dones.all()

    async def read_stdout(self):
        out = await read_async(self.reader)
        return np.frombuffer(out, dtype=np.int8)

    async def recvenv(self) -> Observation:
        arr = await self.read_stdout()
        self.observation = Observation(arr)

    async def reset(self):
        self.reset_env_vars()
        await self.recvenv()
        state, *_, player_index = self.process_state()
        return state, player_index

    def is_current_player_done(self):
        return self.dones[self.current_player] == 1

    async def step(self, action_index: int):
        is_current_player_done = self.is_current_player_done()

        if self.policy_select == 0:
            if not is_current_player_done:
                self.policy_select = action_index + 1
            else:
                await self.recvenv()
        else:
            if not is_current_player_done:
                action = f"{self.current_player}|{action_index-2}\n"
                self.writer.write(action.encode(ENCODING))
                await self.writer.drain()
            await self.recvenv()
            self.policy_select = 0

        return self.process_state()

    def process_state(self) -> Tuple[Dict[str, np.ndarray], int, bool]:
        reward = np.zeros(1)
        if self.policy_select == 0:
            player_index = self.observation.get_player_index()
            self.current_player = player_index.item()
            done = self.observation.get_done()
            self.dones[self.current_player] |= bool(done)
            state = self.observation.get_state()
            reward = self.observation.get_reward()
            self.history[self.current_player].append(state)
            self.history[self.current_player] = self.history[self.current_player][
                -self._NUM_HISTORY :
            ]
            state_stack = State(
                stacknpad(
                    self.history[self.current_player][-self._NUM_HISTORY :],
                    self._NUM_HISTORY,
                ).copy()
            )
            state = state_stack.dense()
            self.prev_state = state

        self.prev_state["legal"] = self.observation.get_legal_moves(self.policy_select)
        is_done = self.is_done()
        reward = reward.reshape((1,))
        return (
            self.prev_state,
            reward.copy(),  # * is_done,
            is_done,
            self.current_player,
        )


class EnvironmentNoStackSingleStep:
    _NUM_HISTORY = 2

    worker_index: int
    act_fn: Callable
    reset_fn: Callable
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter

    @classmethod
    async def create(cls, worker_index: int, act_fn: Callable, reset_fn: Callable):
        reader, writer = await asyncio.open_unix_connection(SOCKET_PATH)
        self = cls()
        self.worker_index = worker_index
        self.act_fn = act_fn
        self.reset_fn = reset_fn
        self.reader = reader
        self.writer = writer
        self.reset_env_vars()
        return self

    async def run(self):
        while True:
            self.reset_fn()
            obs, player_index = await self.reset()
            reward = np.zeros((1,))
            valid = True
            done = False

            while True:
                action = self.act_fn(obs, reward, valid, player_index)
                obs, reward, done, player_index = await self.step(action)
                valid = not bool(reward)
                if done:
                    self.act_fn(obs, reward, valid, player_index)
                    break

    def reset_env_vars(self):
        self.dones = np.array([0, 0], dtype=bool)
        self.history = {0: [], 1: []}
        self.current_player = 0
        self.reward = np.zeros(1)

    def is_done(self):
        return self.dones.all()

    async def read_stdout(self):
        out = await read_async(self.reader)
        return np.frombuffer(out, dtype=np.int8)

    async def recvenv(self) -> Observation:
        arr = await self.read_stdout()
        self.observation = Observation(arr)

    async def reset(self):
        self.reset_env_vars()
        await self.recvenv()
        state, *_, player_index = self.process_state()
        return state, player_index

    def is_current_player_done(self):
        return self.dones[self.current_player] == 1

    async def step(self, action_index: int):
        is_current_player_done = self.is_current_player_done()

        if not is_current_player_done:
            action = f"{self.current_player}|{action_index}\n"
            self.writer.write(action.encode(ENCODING))
            await self.writer.drain()

        await self.recvenv()

        return self.process_state()

    def process_state(self) -> Tuple[Dict[str, np.ndarray], int, bool]:
        reward = np.zeros(1)

        player_index = self.observation.get_player_index()
        self.current_player = player_index.item()
        done = self.observation.get_done()
        self.dones[self.current_player] |= bool(done)
        state = self.observation.get_state()
        reward = self.observation.get_reward()
        self.history[self.current_player].append(state)
        self.history[self.current_player] = self.history[self.current_player][
            -self._NUM_HISTORY :
        ]
        state_stack = State(
            stacknpad(
                self.history[self.current_player][-self._NUM_HISTORY :],
                self._NUM_HISTORY,
            ).copy()
        )
        state = state_stack.dense()

        state["legal"] = self.observation.get_legal_moves_raw()

        is_done = self.is_done()
        reward = reward.reshape((1,))
        return (
            state,
            reward.copy(),  # * is_done,
            is_done,
            self.current_player,
        )
