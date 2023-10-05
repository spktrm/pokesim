import numpy as np

import pickle
import msgpack
import msgpack_numpy as m

from typing import List, Dict, NamedTuple, Sequence

from pokesim.utils import get_arr
from pokesim.types import TensorType
from pokesim.constants import _NUM_HISTORY
from pokesim.data import (
    POSTIONAL_ENCODING_MATRIX,
    TEAM_OFFSET,
    FIELD_OFFSET,
    BOOSTS_OFFSET,
    HISTORY_OFFSET,
    SIDE_CONDITION_OFFSET,
    TURN_OFFSET,
    VOLATILE_STATUS_OFFSET,
    TURN_SIZE,
    FIELD_SIZE,
    NUM_SPECIES,
    NUM_ITEMS,
    NUM_ABILITIES,
    NUM_MOVES,
    NUM_TYPES,
    MAX_HP,
    NUM_HP_BUCKETS,
    NUM_STATUS,
    NUM_BOOSTS,
)

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


def onehot_encode(data: np.ndarray, num_features: int = None):
    if num_features is None:
        num_features = np.max(data) + 1
    data = data.astype(int)
    og_shape = data.shape[:-1]
    num_samples = np.prod(og_shape)
    flattened_data = data.reshape((num_samples, data.shape[-1]))
    multi_hot_encoded = np.zeros((num_samples, num_features), dtype=np.float32)
    multi_hot_encoded[np.arange(num_samples)[:, None], flattened_data] = 1
    return multi_hot_encoded.reshape((*og_shape, num_features))


class State(NamedTuple):
    raw: np.ndarray

    def get_turn(self, leading_dims: Sequence[int]):
        return POSTIONAL_ENCODING_MATRIX[
            self.raw[..., TURN_OFFSET:TEAM_OFFSET]
        ].reshape(*leading_dims, -1)

    def get_teams(self, leading_dims: Sequence[int]):
        teams = np.frombuffer(
            self.raw[..., TEAM_OFFSET:SIDE_CONDITION_OFFSET].tobytes(), dtype=np.int16
        ).reshape(*leading_dims, 3, 6, -1)
        teamsp1 = teams + 1
        species_onehot = onehot_encode(teamsp1[..., 0, None], NUM_SPECIES + 1)[..., 1:]
        item_onehot = onehot_encode(teamsp1[..., 1, None], NUM_ITEMS + 1)[..., 1:]
        ability_onehot = onehot_encode(teamsp1[..., 2, None], NUM_ABILITIES + 1)[
            ..., 1:
        ]
        moves_onehot = onehot_encode(teamsp1[..., -4:], NUM_MOVES + 1)[..., 1:]
        hp_value = teams[..., 3, None] / MAX_HP
        hp_onehot = onehot_encode(np.sqrt(teams[..., 3, None]), NUM_HP_BUCKETS)[..., 1:]
        active = teams[..., 4, None]
        fainted = teams[..., 5, None]
        status_onehot = onehot_encode(teamsp1[..., 6, None], NUM_STATUS + 1)[..., 1:]
        return np.concatenate(
            (
                species_onehot,
                item_onehot,
                ability_onehot,
                moves_onehot,
                hp_value,
                hp_onehot,
                active,
                fainted,
                status_onehot,
            ),
            axis=-1,
        )

    def get_side_conditions(self, leading_dims: Sequence[int]):
        side_conditions = self.raw[
            ..., SIDE_CONDITION_OFFSET:VOLATILE_STATUS_OFFSET
        ].reshape(*leading_dims, 2, -1)
        return side_conditions

    def get_volatile_status(self, leading_dims: Sequence[int]):
        volatile_status = self.raw[..., VOLATILE_STATUS_OFFSET:BOOSTS_OFFSET].reshape(
            *leading_dims, 2, -1
        )
        return volatile_status

    def get_boosts(self, leading_dims: Sequence[int]):
        boosts = self.raw[..., BOOSTS_OFFSET:FIELD_OFFSET].reshape(*leading_dims, 2, -1)
        return boosts

    def get_field(self, leading_dims: Sequence[int]):
        field = self.raw[..., FIELD_OFFSET:HISTORY_OFFSET].reshape(
            *leading_dims, FIELD_SIZE
        )
        return field

    def get_history(self, leading_dims: Sequence[int]):
        history = self.raw[..., HISTORY_OFFSET:].reshape(*leading_dims, -1, 2)
        return history

    def dense(self):
        leading_dims = self.raw.shape[:-1]
        turn_enc = self.get_turn(leading_dims)
        teams_enc = self.get_teams(leading_dims)
        side_conditions_enc = self.get_side_conditions(leading_dims)
        volatile_status_enc = self.get_volatile_status(leading_dims)
        boosts_enc = self.get_boosts(leading_dims)
        field_enc = self.get_field(leading_dims)
        history_enc = self.get_history(leading_dims)
        return {
            "turn": turn_enc,
            "teams": teams_enc,
            "side_conditions": side_conditions_enc,
            "volatile_status": volatile_status_enc,
            "boosts": boosts_enc,
            "field": field_enc,
            "history": history_enc,
        }


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
