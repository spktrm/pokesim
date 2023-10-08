import numpy as np

import pickle
import msgpack
import msgpack_numpy as m

from typing import List, Dict, NamedTuple, Sequence

from pokesim.types import TensorType
from pokesim.data import (
    NUM_PSEUDOWEATHER,
    NUM_TERRAIN,
    NUM_VOLATILE_STATUS,
    NUM_WEATHER,
    POSTIONAL_ENCODING_MATRIX,
    TEAM_OFFSET,
    FIELD_OFFSET,
    BOOSTS_OFFSET,
    HISTORY_OFFSET,
    SIDE_CONDITION_OFFSET,
    TURN_MAX,
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


class TimeStep(NamedTuple):
    id: str
    actor: ActorStep
    env: EnvStep


actor_fields = set(ActorStep._fields)
env_fields = {"player_id", "state", "rewards", "valid", "legal", "history_mask"}
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
    def from_trajectories(cls, batch: List[Trajectory]) -> "Batch":
        store = {k: [] for k in Trajectory._fields}
        for trajectory in batch:
            for key, values in trajectory._asdict().items():
                store[key].append(values)

        max_size = max(store["valid"], key=lambda x: x.shape[0]).shape[0]

        data = {
            key: np.stack(
                [np.resize(sv, (max_size, *sv.shape[1:])) for sv in value], axis=1
            )
            for key, value in store.items()
        }

        return cls(**data)


class OneHotCache:
    def __init__(self):
        self.species = np.eye(NUM_SPECIES + 1, dtype=np.float32)[..., 1:]
        self.items = np.eye(NUM_ITEMS + 1, dtype=np.float32)[..., 1:]
        self.abilities = np.eye(NUM_ABILITIES + 1, dtype=np.float32)[..., 1:]
        self.moves = np.eye(NUM_MOVES + 1, dtype=np.float32)[..., 1:]
        self.hp_buckets = np.eye(NUM_HP_BUCKETS + 1, dtype=np.float32)[..., 1:]
        self.status = np.eye(NUM_STATUS + 1, dtype=np.float32)[..., 1:]
        self.spikes = np.eye(4, dtype=np.float32)[..., 1:]
        self.tspikes = np.eye(3, dtype=np.float32)[..., 1:]
        self.volatile_status = np.eye(NUM_VOLATILE_STATUS + 1, dtype=np.float32)
        self.boosts = np.eye(13, dtype=np.float32)
        self.pseudoweathers = np.eye(NUM_PSEUDOWEATHER + 1, dtype=np.float32)[..., 1:]
        self.weathers = np.eye(NUM_WEATHER + 1, dtype=np.float32)[..., 1:]
        self.terrain = np.eye(NUM_TERRAIN + 1, dtype=np.float32)[..., 1:]

    def onehot_encode(self, data: np.ndarray, feature: str) -> np.ndarray:
        try:
            encodings = getattr(self, feature)
            return encodings[data]
        except Exception as e:
            print(feature)


onehot_cache = OneHotCache()


class State(NamedTuple):
    raw: np.ndarray

    def get_turn(self, leading_dims: Sequence[int]):
        return POSTIONAL_ENCODING_MATRIX[
            self.raw[..., TURN_OFFSET:TEAM_OFFSET].clip(max=TURN_MAX)
        ].reshape(*leading_dims, -1)

    def view_teams(self, leading_dims: Sequence[int]):
        teams = self.raw[..., TEAM_OFFSET:SIDE_CONDITION_OFFSET].view(np.int16)
        return teams.reshape(*leading_dims, 3, 6, -1)

    def get_teams(self, leading_dims: Sequence[int]):
        teams = self.view_teams(leading_dims)
        teamsp1 = teams + 1
        species_onehot = onehot_cache.onehot_encode(teamsp1[..., 0], "species")
        item_onehot = onehot_cache.onehot_encode(teamsp1[..., 1], "items")
        ability_onehot = onehot_cache.onehot_encode(teamsp1[..., 2], "abilities")
        moves_onehot = onehot_cache.onehot_encode(teamsp1[..., -4:], "moves").sum(-2)
        hp_value = (teams[..., 3, None] / MAX_HP).astype(np.float32)
        hp_bucket = np.sqrt(teams[..., 3]).astype(int)
        hp_onehot = onehot_cache.onehot_encode(hp_bucket, "hp_buckets")
        active = teams[..., 4, None].astype(np.float32)
        fainted = teams[..., 5, None].astype(np.float32)
        status_onehot = onehot_cache.onehot_encode(teamsp1[..., 6], "status")
        active_moveset = teamsp1[..., -1, 0, 0, -4:].astype(int)

        entity_encodings = np.concatenate(
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

        return active_moveset, entity_encodings

    def get_side_conditions(self, leading_dims: Sequence[int]):
        side_conditions = self.raw[
            ..., SIDE_CONDITION_OFFSET:VOLATILE_STATUS_OFFSET
        ].reshape(*leading_dims, 2, -1)

        other = side_conditions > 0
        spikes = onehot_cache.onehot_encode(side_conditions[..., 9], "spikes")
        tspikes = onehot_cache.onehot_encode(side_conditions[..., 13], "tspikes")

        side_conditions = np.concatenate((other, spikes, tspikes), axis=-1)
        return side_conditions.reshape(*leading_dims, -1).astype(np.float32)

    def get_volatile_status(self, leading_dims: Sequence[int]):
        volatile_status = self.raw[..., VOLATILE_STATUS_OFFSET:BOOSTS_OFFSET].reshape(
            *leading_dims, 2, 2, -1
        )
        volatile_status_id = volatile_status[..., 0, :]
        volatile_status_level = volatile_status[..., 1, :]
        volatile_status = onehot_cache.onehot_encode(
            volatile_status_id + 1, "volatile_status"
        ).sum(-2)
        return volatile_status.reshape(*leading_dims, -1).astype(np.float32)

    def get_boosts(self, leading_dims: Sequence[int]):
        boosts = self.raw[..., BOOSTS_OFFSET:FIELD_OFFSET].reshape(*leading_dims, 2, -1)
        boosts_onehot = onehot_cache.onehot_encode(boosts + 6, "boosts")
        boosts_onehot = np.concatenate((boosts_onehot[..., :6], boosts_onehot[..., 7:]))
        boosts_scaled = np.sign(boosts) * np.sqrt(abs(boosts))
        boosts = np.concatenate(
            (
                boosts_onehot.reshape(*leading_dims, -1),
                boosts_scaled.reshape(*leading_dims, -1),
            ),
            axis=-1,
        )
        return boosts.astype(np.float32)

    def get_field(self, leading_dims: Sequence[int]):
        field = self.raw[..., FIELD_OFFSET:HISTORY_OFFSET].reshape(*leading_dims, 3, 5)
        field_id = field[..., 0, :]
        pseudoweathers = field_id[..., :3]
        pseudoweathers_onehot = onehot_cache.onehot_encode(
            pseudoweathers + 1, "pseudoweathers"
        ).sum(-2)
        weather = field_id[..., 3]
        weather_onehot = onehot_cache.onehot_encode(weather + 1, "weathers")
        terrain = field_id[..., 4]
        terrain_onehot = onehot_cache.onehot_encode(terrain + 1, "terrain")
        field_min_durr = field[..., 1, :]
        field_max_durr = field[..., 2, :]
        field = np.concatenate(
            (pseudoweathers_onehot, weather_onehot, terrain_onehot), axis=-1
        )
        return field.reshape(*leading_dims, -1).astype(np.float32)

    def _get_history(self, leading_dims: Sequence[int]):
        teams = np.frombuffer(
            self.raw[..., HISTORY_OFFSET:].tobytes(),
            dtype=np.int16,
        )
        return teams.reshape(*leading_dims, -1, 2)

    def get_history(self, leading_dims: Sequence[int]):
        history = self._get_history(leading_dims)
        return history.astype(np.int64)

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
