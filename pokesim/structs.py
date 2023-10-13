import numpy as np

import pickle
import msgpack
import msgpack_numpy as m

from typing import List, Dict, NamedTuple, Sequence

from pokesim.types import TensorType
from pokesim.data import (
    POSTIONAL_ENCODING_MATRIX,
    TEAM_OFFSET,
    FIELD_OFFSET,
    BOOSTS_OFFSET,
    HISTORY_OFFSET,
    SIDE_CONDITION_OFFSET,
    TURN_MAX,
    TURN_OFFSET,
    VOLATILE_STATUS_OFFSET,
)


class EnvStep(NamedTuple):
    game_id: np.ndarray
    player_id: np.ndarray
    state: np.ndarray
    rewards: np.ndarray
    valid: np.ndarray
    legal: np.ndarray
    history_mask: np.ndarray


class ModelOutput(NamedTuple):
    policy: TensorType
    log_policy: TensorType
    logits: TensorType
    value: TensorType


class ActorStep(NamedTuple):
    policy: np.ndarray
    action: np.ndarray
    rewards: np.ndarray


class TimeStep(NamedTuple):
    id: str
    actor: ActorStep
    env: EnvStep


actor_fields = set(ActorStep._fields)
env_fields = {"player_id", "state", "valid", "legal", "history_mask"}
_FIELDS_TO_STORE = actor_fields | env_fields


class Trajectory(NamedTuple):
    # Env fields
    player_id: TensorType
    state: TensorType
    rewards: TensorType
    valid: TensorType
    legal: TensorType
    history_mask: TensorType

    # Actor fields
    policy: TensorType
    action: TensorType

    def __len__(self):
        return max(self.valid.sum(0, keepdims=True))

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
            **{key: np.stack(value) for key, value in store.items()},
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
    def from_trajectories(cls, traj_list: List[Trajectory]) -> "Batch":
        lengths = np.array([len(t) for t in traj_list])
        max_index = lengths.argmax(-1)
        batch_size = len(traj_list)

        store = {}
        for k in Trajectory._fields:
            value = getattr(traj_list[max_index], k)[:, None]
            new_shape = (1, batch_size, *((1,) * len(value.shape[2:])))
            store[k] = np.tile(value, new_shape)

        for batch_index, trajectory in enumerate(traj_list):
            if batch_index == max_index:
                continue
            trajectory_length = len(trajectory)
            for key, values in trajectory._asdict().items():
                store[key][:trajectory_length, batch_index] = values
            store["valid"][trajectory_length:, batch_index] = False
            store["rewards"][trajectory_length:, batch_index] *= 0

        return cls(**store)


class State(NamedTuple):
    raw: np.ndarray

    def get_turn(self, leading_dims: Sequence[int]):
        return POSTIONAL_ENCODING_MATRIX[
            self.raw[..., TURN_OFFSET:TEAM_OFFSET].clip(max=TURN_MAX)
        ].reshape(*leading_dims, -1)

    def view_teams(self, leading_dims: Sequence[int]):
        teams = self.raw[..., TEAM_OFFSET:SIDE_CONDITION_OFFSET].view(np.int16)
        return teams.reshape(*leading_dims, 3, 6, -1).astype(int)

    def get_teams(self, leading_dims: Sequence[int]):
        teams = self.view_teams(leading_dims)
        active_moveset = teams[..., -1, 0, 0, -4:].astype(int) + 2
        return active_moveset, teams

    def get_side_conditions(self, leading_dims: Sequence[int]):
        return (
            self.raw[..., SIDE_CONDITION_OFFSET:VOLATILE_STATUS_OFFSET]
            .reshape(*leading_dims, 2, -1)
            .astype(int)
        )

    def get_volatile_status(self, leading_dims: Sequence[int]):
        return (
            self.raw[..., VOLATILE_STATUS_OFFSET:BOOSTS_OFFSET]
            .reshape(*leading_dims, 2, -1, 2)
            .astype(int)
        )

    def get_boosts(self, leading_dims: Sequence[int]):
        return (
            self.raw[..., BOOSTS_OFFSET:FIELD_OFFSET]
            .reshape(*leading_dims, 2, -1)
            .astype(int)
        )

    def get_field(self, leading_dims: Sequence[int]):
        return (
            self.raw[..., FIELD_OFFSET:HISTORY_OFFSET]
            .reshape(*leading_dims, 5, 3)
            .astype(int)
        )

    def _get_history(self, leading_dims: Sequence[int]):
        teams = np.frombuffer(
            self.raw[..., HISTORY_OFFSET:].tobytes(),
            dtype=np.int16,
        )
        return teams.reshape(*leading_dims, -1, 2)

    def get_history(self, leading_dims: Sequence[int]):
        history = self._get_history(leading_dims)
        return history.astype(int)

    def dense(self):
        leading_dims = self.raw.shape[:-1]
        turn_enc = self.get_turn(leading_dims)
        active_moveset_enc, teams_enc = self.get_teams(leading_dims)
        side_conditions_enc = self.get_side_conditions(leading_dims)
        volatile_status_enc = self.get_volatile_status(leading_dims)
        boosts_enc = self.get_boosts(leading_dims)
        field_enc = self.get_field(leading_dims)
        history_enc = self.get_history(leading_dims)
        return {
            "raw": self.raw,
            "turn": turn_enc,
            "active_moveset": active_moveset_enc,
            "teams": teams_enc,
            "side_conditions": side_conditions_enc,
            "volatile_status": volatile_status_enc,
            "boosts": boosts_enc,
            "field": field_enc,
            "history": history_enc,
        }


class Observation(NamedTuple):
    raw: np.ndarray

    def get_state(self):
        return self.raw[..., 4:-10]

    def get_legal_moves(self, policy_select: np.ndarray):
        mask = self.raw[..., -10:]
        move_mask = mask[..., :4].copy()
        switch_mask = mask[..., 4:].copy()
        action_type_mask = np.concatenate(
            (
                move_mask.any(-1, keepdims=True),
                switch_mask.any(-1, keepdims=True),
            ),
            axis=-1,
        )
        move_mask += ~action_type_mask[..., 0, None]
        switch_mask += ~action_type_mask[..., 1, None]
        mask = np.concatenate(
            (
                action_type_mask * (policy_select == 0),
                move_mask * (policy_select == 1),
                switch_mask * (policy_select == 2),
            ),
            axis=-1,
        )
        return mask.astype(np.bool_)

    def get_worker_index(self):
        return self.raw[..., 0]

    def get_player_index(self):
        return self.raw[..., 1]

    def get_done(self):
        return self.raw[..., 2]

    def get_reward(self):
        return self.raw[..., 3]
