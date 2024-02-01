import torch
import numpy as np

import pickle
import msgpack
import msgpack_numpy as m

from typing import List, Dict, NamedTuple, Sequence, Tuple

from pokesim.data import (
    HEURISTIC_OFFSET,
    HISTORY_BOOSTS_OFFSET,
    HISTORY_FIELD_OFFSET,
    HISTORY_SIDE_CONDITION_OFFSET,
    HISTORY_VOLATILE_STATUS_OFFSET,
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
    history_mask: np.ndarray = None


class ModelOutput(NamedTuple):
    policy: torch.Tensor
    log_policy: torch.Tensor
    logits: torch.Tensor
    value: torch.Tensor

    def to_flat(self, order: torch.Tensor, max_index: int):
        VALID_KEYS = {"policy", "log_policy", "logits", "value"}
        return ModelOutput(
            **{
                k: torch.take_along_dim(
                    torch.cat((v[:, ::2], v[:, 1::2])),
                    order.reshape(order.shape + ((1,) * (len(v.shape) - 2))),
                    0,
                )[:max_index]
                for k, v in self._asdict().items()
                if k in VALID_KEYS
            }
        )


class ActorStep(NamedTuple):
    policy: np.ndarray
    action: np.ndarray
    rewards: np.ndarray
    value: np.ndarray


class TimeStep(NamedTuple):
    id: str
    actor: ActorStep
    env: EnvStep
    ts: int


actor_fields = set(ActorStep._fields)
env_fields = {"game_id", "player_id", "state", "valid", "legal"}
_FIELDS_TO_STORE = actor_fields | env_fields | {"ts"}


class Trajectory(NamedTuple):
    # Env fields
    game_id: np.ndarray
    player_id: np.ndarray
    state: np.ndarray
    rewards: np.ndarray
    valid: np.ndarray
    legal: np.ndarray

    # Actor fields
    policy: np.ndarray
    action: np.ndarray
    value: np.ndarray
    ts: np.ndarray

    def get_length(self):
        return max(self.valid.sum(0, keepdims=True))  # + 1

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
        for _, actor_step, env_step, ts in traj:
            for key in actor_fields:
                value = getattr(actor_step, key)
                if value is not None:
                    store[key].append(value)

            for key in env_fields:
                value = getattr(env_step, key)
                if value is not None:
                    store[key].append(value)

            store["ts"].append(ts)

        elements = {key: np.stack(value) for key, value in store.items() if value}

        return cls(**elements)

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
    def from_trajectories(cls, traj_list: Tuple[Trajectory]) -> "Batch":
        lengths = np.array([t.get_length() for t in traj_list])
        max_index = lengths.argmax(-1)
        batch_size = len(traj_list)

        store = {}
        for k in Trajectory._fields:
            value = getattr(traj_list[max_index], k)
            if value is None:
                continue

            value = value[:, None]
            new_shape = (1, batch_size, *((1,) * len(value.shape[2:])))
            store[k] = np.tile(value, new_shape)

        # traj_list = [traj_list[i] for i in np.argsort(lengths)]

        for batch_index, trajectory in enumerate(traj_list):
            trajectory_length = trajectory.get_length()
            for key, values in trajectory._asdict().items():
                if values is not None:
                    store[key][:trajectory_length, batch_index] = values[
                        :trajectory_length
                    ]
            store["valid"][trajectory_length:, batch_index] = False
            store["rewards"][trajectory_length:, batch_index] = 0
            store["ts"][trajectory_length:, batch_index] = int(1e4)

        return cls(**store)


class State(NamedTuple):
    raw: np.ndarray

    def get_turn(self, leading_dims: Sequence[int]):
        return (
            self.raw[..., TURN_OFFSET:HEURISTIC_OFFSET]
            .clip(max=TURN_MAX)
            .reshape(*leading_dims)
            .astype(int)
        )

    def get_heuristic_action(self, leading_dims: Sequence[int]):
        return (
            self.raw[..., HEURISTIC_OFFSET:TEAM_OFFSET]
            .reshape(*leading_dims)
            .astype(int)
        )

    def view_teams(self, leading_dims: Sequence[int]):
        teams = self.raw[..., TEAM_OFFSET:SIDE_CONDITION_OFFSET].view(np.int16)
        return teams.reshape(*leading_dims, 3, 6, -1).astype(int)

    def get_teams(self, leading_dims: Sequence[int]):
        teams = self.view_teams(leading_dims)
        return teams

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
        history = self.raw[..., HISTORY_OFFSET:].view(np.int8)
        history = history.reshape(*leading_dims, -1, 213)
        history_right = history[..., 99:].view(np.int16)
        return history[..., :99], history_right[..., :-9], history_right[..., -9:]

    def get_history(self, leading_dims: Sequence[int]):
        history_context, history_entities, history_stats = self._get_history(
            leading_dims
        )
        return {
            "history_side_conditions": history_context[
                ..., HISTORY_SIDE_CONDITION_OFFSET:HISTORY_VOLATILE_STATUS_OFFSET
            ]
            .reshape(*leading_dims, 20, 2, -1)
            .astype(int),
            "history_volatile_status": history_context[
                ..., HISTORY_VOLATILE_STATUS_OFFSET:HISTORY_BOOSTS_OFFSET
            ]
            .reshape(*leading_dims, 20, 2, -1, 2)
            .astype(int),
            "history_boosts": history_context[
                ..., HISTORY_BOOSTS_OFFSET:HISTORY_FIELD_OFFSET
            ]
            .reshape(*leading_dims, 20, 2, -1)
            .astype(int),
            "history_field": history_context[..., HISTORY_FIELD_OFFSET:]
            .reshape(*leading_dims, 20, 5, 3)
            .astype(int),
            "history_entities": history_entities.reshape(
                *history_entities.shape[:-1], 2, -1
            ).astype(int),
            "history_stats": history_stats.astype(int),
        }

    def dense(self):
        if len(self.raw.shape) > 2:
            leading_dims = self.raw.shape[:-1]
        else:
            leading_dims = (1, 1) + (self.raw.shape[0],)
        turn_enc = self.get_turn(leading_dims)
        heuristic_action = self.get_heuristic_action(leading_dims)
        teams_enc = self.get_teams(leading_dims)
        side_conditions_enc = self.get_side_conditions(leading_dims)
        volatile_status_enc = self.get_volatile_status(leading_dims)
        boosts_enc = self.get_boosts(leading_dims)
        field_enc = self.get_field(leading_dims)
        history_enc = self.get_history(leading_dims)
        return {
            "raw": self.raw,
            "turn": turn_enc,
            "heuristic_action": heuristic_action,
            "teams": teams_enc,
            "side_conditions": side_conditions_enc,
            "volatile_status": volatile_status_enc,
            "boosts": boosts_enc,
            "field": field_enc,
            **history_enc,
        }


class Observation(NamedTuple):
    raw: np.ndarray

    def get_state(self):
        return self.raw[..., 4:-10]

    def get_legal_moves_raw(self):
        return self.raw[..., -10:].copy().astype(np.bool_)

    def get_legal_moves(self, policy_select: np.ndarray):
        mask = self.raw[..., -10:].copy()
        move_mask = mask[..., :4]
        switch_mask = mask[..., 4:]
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
