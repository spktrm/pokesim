import asyncio
import struct
import numpy as np

from typing import Callable, Sequence, Tuple, Dict

from pokesim.data import SOCKET_PATH, ENCODING, NUM_HISTORY
from pokesim.structs import Observation, State


async def read_async(sock: asyncio.StreamReader) -> bytes:
    state_size_buffer = await sock.read(4)
    state_size = struct.unpack("<I", state_size_buffer)[0]
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


class EnvironmentNoStackSingleStep:
    _NUM_HISTORY = 2

    worker_index: int
    act_fn: Callable
    reset_fn: Callable
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    threshold: int

    @classmethod
    async def create(cls, worker_index: int, threshold: int = 2):
        reader, writer = await asyncio.open_unix_connection(SOCKET_PATH)
        self = cls()
        self.worker_index = worker_index
        self.reader = reader
        self.writer = writer
        self.threshold = threshold
        self.reset_env_vars()
        return self

    async def run(self):
        while True:
            obs, player_index = await self.reset()
            reward = 0

            while True:
                action = self.act_fn(
                    obs, reward, self.dones[player_index], player_index
                )
                state = await self.step(action)
                obs, reward, done, player_index = state
                self.dones[player_index] |= done

                if self.dones.all():
                    self.reset_fn()

    def reset_env_vars(self):
        self.dones = np.array([0, 0], dtype=bool)
        self.history = {0: [], 1: []}
        self.current_player = 0
        self.reward = np.zeros(1)

    def is_done(self):
        if self.threshold > 1:
            return self.dones.all()
        else:
            return self.dones.any()

    async def read_stdout(self):
        out = await read_async(self.reader)
        return np.frombuffer(out, dtype=np.int8)

    async def recvenv(self) -> Observation:
        arr = await self.read_stdout()
        self.observation = Observation(arr)

    async def reset(self):
        self.reset_env_vars()
        await self.recvenv()
        return self.process_state()

    async def step(self, action_index: int):
        action = f"{self.current_player}|{action_index}\n"
        self.writer.write(action.encode(ENCODING))
        await self.writer.drain()
        await self.recvenv()
        return self.process_state()

    def process_state(self) -> Tuple[Dict[str, np.ndarray], int, bool, int]:
        reward = np.zeros(1)

        worker_index = self.observation.get_worker_index()
        player_index = self.observation.get_player_index()
        self.current_player = player_index.item()
        done = self.observation.get_done()
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

        state["legal"] = self.observation.get_legal_moves()

        reward = reward.reshape((1,))
        return (
            worker_index,
            state,
            reward.copy(),  # * is_done,
            bool(done),
            self.current_player,
        )
