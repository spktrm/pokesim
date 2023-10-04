import numpy as np

import pickle
import msgpack
import msgpack_numpy as m

from typing import List, Dict, NamedTuple

from pokesim.utils import get_arr
from pokesim.types import TensorType
from pokesim.constants import _NUM_HISTORY

_r = lambda arr: arr.reshape(1, 1, -1)


class EnvStep(NamedTuple):
    game_id: np.ndarray
    player_id: np.ndarray
    raw_obs: np.ndarray
    rewards: np.ndarray
    valid: np.ndarray
    legal: np.ndarray

    @classmethod
    def from_stack(cls, env_steps: List["EnvStep"], pad_depth: int = _NUM_HISTORY):
        latest = env_steps[-1]
        stacked = np.stack([step.raw_obs for step in env_steps], axis=2)
        if stacked.shape[2] < pad_depth:
            pad_shape = list(stacked.shape)
            pad_shape[2] = pad_depth - stacked.shape[2]
            stacked = np.concatenate(
                (stacked, np.zeros(shape=pad_shape, dtype=stacked.dtype)), axis=2
            )
        return cls(
            game_id=latest.game_id,
            player_id=latest.player_id,
            raw_obs=stacked,
            rewards=latest.rewards,
            valid=latest.valid,
            legal=latest.legal,
        )

    @classmethod
    def from_data(cls, data: bytes) -> "EnvStep":
        state = _r(get_arr(data[:-2]))
        legal = state[..., -10:].astype(bool)
        valid = (1 - state[..., 3]).astype(bool)
        player_id = state[..., 2]
        winner = state[..., 4]
        if winner >= 0:
            rew = 2 * int((player_id == winner).item()) - 1
            rewards = np.array([rew, -rew])
        else:
            rewards = np.array([0, 0])
        rewards = _r(rewards)
        if player_id == 1:
            rewards = np.flip(rewards, axis=-1)
        return cls(
            game_id=state[..., 1],
            player_id=player_id,
            raw_obs=state[..., 6:-10],
            rewards=rewards,
            valid=valid,
            legal=legal,
        )

    @classmethod
    def from_prev_and_curr(cls, prev: "EnvStep", curr: "EnvStep") -> "EnvStep":
        return cls(
            game_id=prev.game_id,
            player_id=prev.player_id,
            raw_obs=prev.raw_obs,
            rewards=curr.rewards,
            valid=prev.valid,
            legal=prev.legal,
        )

    def get_leading_dims(self):
        return self.valid.shape


class ModelOutput(NamedTuple):
    policy: TensorType
    log_policy: TensorType
    logits: TensorType
    value: TensorType


class ActorStep(NamedTuple):
    policy: np.ndarray
    action: np.ndarray
    policy_select: np.ndarray


class TimeStep(NamedTuple):
    id: str
    actor: ActorStep
    env: EnvStep


_DTYPES = {
    "player_id": np.int16,
    "raw_obs": np.int16,
    "rewards": np.int64,
    "valid": np.bool_,
    "legal": np.bool_,
    "policy": np.float32,
    "action": np.int64,
}

actor_fields = set(ActorStep._fields)
env_fields = {"player_id", "raw_obs", "rewards", "valid", "legal"}
_FIELDS_TO_STORE = actor_fields | env_fields


class Trajectory(NamedTuple):
    # Env fields
    player_id: TensorType
    raw_obs: TensorType
    rewards: TensorType
    valid: TensorType
    legal: TensorType

    # Actor fields
    policy: TensorType
    action: TensorType

    # policy select
    policy_select: TensorType

    def __len__(self):
        return self.valid.sum()

    def save(self, fpath: str):
        print(f"Saving `{fpath}`")
        with open(fpath, "wb") as f:
            f.write(pickle.dumps(self.serialize()))

    @classmethod
    def load(cls, fpath: str):
        with open(fpath, "rb") as f:
            data = pickle.loads(f.read())
        return Trajectory.deserialize(data)

    def is_valid(self):
        return self.valid.sum() > 0

    @classmethod
    def from_env_steps(cls, traj: List[TimeStep]) -> "Trajectory":
        store = {k: [] for k in _FIELDS_TO_STORE}
        for _, actor_step, env_step in traj:
            for key in actor_fields:
                store[key].append(getattr(actor_step, key))
            for key in env_fields:
                store[key].append(getattr(env_step, key))
        return cls(
            **{key: np.concatenate(value) for key, value in store.items()},
        )

    def serialize(self):
        return {
            k: msgpack.packb(v, default=m.encode) for k, v in self._asdict().items()
        }

    @classmethod
    def deserialize(cls, data: Dict[str, bytes]):
        return cls(
            **{k: msgpack.unpackb(v, object_hook=m.decode) for k, v in data.items()}
        )


class Batch(Trajectory):
    @classmethod
    def from_trajectories(cls, batch: List[Trajectory], sort: bool = True) -> "Batch":
        store = {k: [] for k in Trajectory._fields}
        for trajectory in batch:
            for key, values in trajectory._asdict().items():
                store[key].append(values)

        max_size = max(store["valid"], key=lambda x: x.shape[0]).shape[0]

        data = {
            key: np.concatenate(
                [np.resize(sv, (max_size, *sv.shape[1:])) for sv in value], axis=1
            )
            for key, value in store.items()
        }
        arange = np.arange(data["valid"].shape[0])[:, None]
        amax = np.argmax(data["valid"] == False, 0)
        valid = arange < amax
        data["valid"] = valid
        data["rewards"][:-1] = data["rewards"][1:]
        _rewards_prev = data["rewards"]
        data["rewards"] = _rewards_prev * (arange == (amax - 1))[..., None]

        if sort:
            order = np.argsort(valid.sum(0))
            data = {
                key: np.ascontiguousarray(value[:, order])
                for key, value in data.items()
            }

        return cls(**data)
