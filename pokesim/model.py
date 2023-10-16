import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence

from typing import Sequence
from pokesim.data import (
    NUM_ABILITIES,
    NUM_BOOSTS,
    NUM_HP_BUCKETS,
    NUM_ITEMS,
    NUM_MOVES,
    NUM_PSEUDOWEATHER,
    NUM_SIDE_CONDITIONS,
    NUM_SPECIES,
    NUM_STATUS,
    NUM_TERRAIN,
    NUM_VOLATILE_STATUS,
    NUM_WEATHER,
    TURN_ENC_SIZE,
)

from pokesim.structs import ModelOutput
from pokesim.utils import _legal_log_policy, _legal_policy

_USE_LAYER_NORM = False


def _layer_init(
    layer: nn.Module, mean: float = None, std: float = None, bias_value: float = None
):
    if hasattr(layer, "weight"):
        if isinstance(layer, nn.Embedding):
            init_func = nn.init.normal_
        elif isinstance(layer, nn.Linear):
            init_func = nn.init.trunc_normal_
        if std is None:
            n = getattr(layer, "num_embeddings", None) or getattr(layer, "in_features")
            std = 1 / math.sqrt(n)
        init_func(layer.weight, mean=(mean or 0), std=std)
    if hasattr(layer, "bias") and getattr(layer, "bias", None) is not None:
        nn.init.constant_(layer.bias, val=(bias_value or 0))
    return layer


class ResBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = None,
        output_size: int = None,
        num_layers: int = 2,
        bias: bool = True,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()

        output_size = output_size or input_size
        hidden_size = hidden_size or input_size
        sizes = (
            [input_size]
            + [hidden_size for _ in range(max(0, num_layers - 1))]
            + [output_size]
        )
        layers = []
        for size1, size2 in zip(sizes, sizes[1:]):
            layer = [
                nn.ReLU(),
                _layer_init(nn.Linear(size1, size2, bias=bias)),
            ]
            if use_layer_norm:
                layer.insert(0, nn.LayerNorm(size1))
            layers += layer
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + x


class ResNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = None,
        output_size: int = None,
        num_resblocks: int = 2,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.resblocks = nn.ModuleList(
            [
                ResBlock(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size,
                    use_layer_norm=use_layer_norm,
                )
                for i in range(num_resblocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resblock in self.resblocks:
            x = resblock(x)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        layer_sizes: Sequence[int] = None,
        bias: bool = True,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.layer_sizes = layer_sizes
        layers = []
        for size1, size2 in zip(layer_sizes, layer_sizes[1:]):
            layer = [
                nn.ReLU(),
                _layer_init(nn.Linear(size1, size2, bias=bias)),
            ]
            if use_layer_norm:
                layer.insert(0, nn.LayerNorm(size1))
            layers += layer
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        key_size: int,
        with_bias: bool = True,
        value_size: int = None,
        model_size: int = None,
    ):
        super().__init__()
        self.key_size = key_size
        self.num_heads = num_heads
        self.value_size = value_size or key_size
        self.model_size = model_size or key_size * num_heads
        self.denom = 1 / math.sqrt(key_size)

        self.queries = _layer_init(
            nn.Linear(self.model_size, num_heads * self.key_size, bias=with_bias)
        )
        self.keys = _layer_init(
            nn.Linear(self.model_size, num_heads * self.key_size, bias=with_bias)
        )
        self.values = _layer_init(
            nn.Linear(self.model_size, num_heads * self.value_size, bias=with_bias)
        )
        self.final_proj = _layer_init(
            nn.Linear(self.value_size * num_heads, self.model_size)
        )

    def _linear_projection(
        self, x: torch.Tensor, mod: nn.Module, head_size: int
    ) -> torch.Tensor:
        y = mod(x)
        *leading_dims, _ = x.shape
        return y.reshape((*leading_dims, self.num_heads, head_size))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        *leading_dims, sequence_length, _ = query.shape

        query_heads = self._linear_projection(query, self.queries, self.key_size)
        key_heads = self._linear_projection(key, self.keys, self.key_size)
        value_heads = self._linear_projection(value, self.values, self.value_size)

        attn_logits = torch.einsum("...thd,...Thd->...htT", query_heads, key_heads)
        attn_logits = attn_logits * self.denom
        if mask is not None:
            if mask.ndim != attn_logits.ndim:
                raise ValueError(
                    f"Mask dimensionality {mask.ndim} must match logits dimensionality "
                    f"{attn_logits.ndim}."
                )
            attn_logits = torch.where(mask, attn_logits, -1e30)
        attn_weights = attn_logits.softmax(-1)

        attn = torch.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
        attn = torch.reshape(attn, (*leading_dims, sequence_length, -1))

        return self.final_proj(attn)


class Transformer(nn.Module):
    def __init__(
        self,
        transformer_num_layers: int,
        transformer_num_heads: int,
        transformer_key_size: int,
        transformer_value_size: int,
        transformer_model_size: int,
        resblocks_num_before: int,
        resblocks_num_after: int,
        resblocks_hidden_size: int = None,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        self.attn = nn.ModuleList(
            [
                MultiHeadAttention(
                    transformer_num_heads,
                    transformer_key_size,
                    value_size=transformer_value_size,
                    model_size=transformer_model_size,
                )
                for i in range(transformer_num_layers)
            ]
        )
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.ln = nn.ModuleList(
                [
                    nn.LayerNorm(transformer_model_size)
                    for _ in range(transformer_num_layers)
                ]
            )
        self.resnet_before = ResNet(
            input_size=transformer_model_size,
            hidden_size=resblocks_hidden_size,
            output_size=transformer_model_size,
            num_resblocks=resblocks_num_before,
            use_layer_norm=use_layer_norm,
        )
        self.resnet_after = ResNet(
            input_size=transformer_model_size,
            hidden_size=resblocks_hidden_size,
            output_size=transformer_model_size,
            num_resblocks=resblocks_num_after,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.resnet_before(x)
        for i, attn in enumerate(self.attn):
            x1 = x
            if self.use_layer_norm:
                ln = self.ln[i]
                x1 = ln(x1)
            x1 = F.relu(x1)
            logits_mask = mask[..., None, None, :]
            x1 = attn(x1, x1, x1, logits_mask)
            x1 = torch.where(mask.unsqueeze(-1), x1, 0)
            x = x + x1
        x = self.resnet_after(x)
        x = torch.where(mask.unsqueeze(-1), x, 0)
        return x


class ToVector(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        use_layer_norm: bool = True,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.net = MLP([input_size] + hidden_sizes, use_layer_norm=use_layer_norm)
        out_layers = [
            nn.ReLU(),
            _layer_init(nn.Linear(hidden_sizes[-1], hidden_sizes[-1])),
        ]
        if use_layer_norm:
            out_layers.insert(0, nn.LayerNorm(hidden_sizes[-1]))
        self.out = nn.Sequential(*out_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x).flatten(-3, -2).mean(-2)
        x = self.out(x)
        return x


class VectorMerge(nn.Module):
    def __init__(
        self, input_sizes: Sequence[int], output_size: int, use_layer_norm: bool = True
    ) -> None:
        super().__init__()

        self.out_lins = nn.ModuleList(
            [
                _layer_init(nn.Linear(input_size, output_size))
                for input_size in input_sizes
            ]
        )
        self.gate_lins = nn.ModuleList(
            [
                _layer_init(nn.Linear(input_size, len(input_sizes) * output_size))
                for input_size in input_sizes
            ]
        )
        self.gate_size = output_size

    def _compute_gate(self, init_gate: Sequence[torch.Tensor]):
        gate = [gate(y) for y, gate in zip(init_gate, self.gate_lins)]
        gate = sum(gate)
        gate = gate.view(*gate.shape[:-1], len(init_gate), self.gate_size)
        gate = gate.softmax(axis=-2)
        return gate

    def _encode(self, inputs: Sequence[torch.Tensor]):
        gate, outputs = [], []
        for input, out_lin in zip(inputs, self.out_lins):
            feature = F.relu(input)
            gate.append(feature)
            outputs.append(out_lin(feature))
        return gate, outputs

    def forward(self, *inputs: Sequence[torch.Tensor]):
        gate, outputs = self._encode(inputs)
        gate = self._compute_gate(gate)
        data = gate * torch.stack(outputs, dim=-2)
        output = data.sum(-2)
        return output


class PointerLogits(nn.Module):
    def __init__(
        self,
        query_input_size: int,
        keys_input_size: int,
        num_layers_query: int = 2,
        num_layers_keys: int = 2,
        key_size: int = 64,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        self.query_mlp = MLP(
            [query_input_size]
            + [query_input_size for _ in range(num_layers_query - 1)]
            + [key_size],
            use_layer_norm=use_layer_norm,
        )
        self.keys_mlp = MLP(
            [keys_input_size]
            + [keys_input_size for _ in range(num_layers_keys - 1)]
            + [key_size],
            use_layer_norm=use_layer_norm,
        )

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        query = self.query_mlp(query)
        keys = self.keys_mlp(keys)

        logits = keys @ query.transpose(-2, -1)
        return logits


class ConvPointerLogits(nn.Module):
    def __init__(self, query_input_size: int, keys_input_size: int, output_size: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(query_input_size + keys_input_size, 256, 3, 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1),
            nn.Flatten(-2),
            MLP([256 * (output_size - 2), 256, 256, output_size]),
        )

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        T, B, *_ = query.shape
        gated_keys = (
            torch.cat((query.expand(*keys.shape[:-1], -1), keys), dim=-1)
            .transpose(-2, -1)
            .flatten(0, 1)
        )
        logits = self.conv(gated_keys)
        return logits.view(T, B, -1)


class Model(nn.Module):
    def __init__(self, entity_size: int = 64, vector_size: int = 256):
        super().__init__()

        side_token = torch.zeros(18, dtype=torch.long).view(-1, 6)
        side_token[-1] = 1
        self.register_buffer("side_token", side_token)

        public_token = torch.zeros(18, dtype=torch.long).view(-1, 6)
        public_token[1:] = 1
        self.register_buffer("public_token", public_token)

        self.species_onehot = _layer_init(nn.Embedding(NUM_SPECIES + 1, entity_size))
        self.item_onehot = _layer_init(nn.Embedding(NUM_ITEMS + 1, entity_size))
        self.ability_onehot = _layer_init(nn.Embedding(NUM_ABILITIES + 1, entity_size))
        self.moves_onehot = _layer_init(nn.Embedding(NUM_MOVES + 2, entity_size))
        self.hp_onehot = _layer_init(nn.Embedding(NUM_HP_BUCKETS + 1, entity_size))
        self.hp_value = _layer_init(nn.Linear(1, entity_size))
        self.status_onehot = _layer_init(nn.Embedding(NUM_STATUS + 1, entity_size))
        self.active_onehot = _layer_init(nn.Embedding(2, entity_size))
        self.fainted_onehot = _layer_init(nn.Embedding(2, entity_size))
        self.side_onehot = _layer_init(nn.Embedding(2, entity_size))
        self.public_onehot = _layer_init(nn.Embedding(2, entity_size))

        # self.hp_onehot = nn.Embedding.from_pretrained(
        #     torch.eye(NUM_HP_BUCKETS + 1)[..., 1:]
        # )
        # self.status_onehot = nn.Embedding.from_pretrained(
        #     torch.eye(NUM_STATUS + 1)[..., 1:]
        # )
        # self.active_onehot = nn.Embedding.from_pretrained(torch.eye(2))
        # self.fainted_onehot = nn.Embedding.from_pretrained(torch.eye(2))
        # self.side_onehot = nn.Embedding.from_pretrained(torch.eye(2))
        # self.public_onehot = nn.Embedding.from_pretrained(torch.eye(2))

        # raw_entity_size = (
        #     self.species_onehot.weight.shape[-1]
        #     + self.item_onehot.weight.shape[-1]
        #     + self.ability_onehot.weight.shape[-1]
        #     + self.moves_onehot.weight.shape[-1]
        #     + self.hp_onehot.weight.shape[-1]
        #     + self.status_onehot.weight.shape[-1]
        #     + self.active_onehot.weight.shape[-1]
        #     + self.fainted_onehot.weight.shape[-1]
        #     + self.side_onehot.weight.shape[-1]
        #     + self.public_onehot.weight.shape[-1]
        #     + 1
        # )
        self.units_mlp = MLP([entity_size, entity_size], use_layer_norm=_USE_LAYER_NORM)

        self.spikes_onehot = nn.Embedding.from_pretrained(torch.eye(4)[..., 1:])
        self.tspikes_onehot = nn.Embedding.from_pretrained(torch.eye(3)[..., 1:])
        self.volatile_status_onehot = nn.Embedding.from_pretrained(
            torch.eye(NUM_VOLATILE_STATUS + 1)[..., 1:]
        )
        self.boosts_onehot = nn.Embedding.from_pretrained(torch.eye(13))
        self.pseudoweathers_onehot = nn.Embedding.from_pretrained(
            torch.eye(NUM_PSEUDOWEATHER + 1)[..., 1:]
        )
        self.weathers_onehot = nn.Embedding.from_pretrained(
            torch.eye(NUM_WEATHER + 1)[..., 1:]
        )
        self.terrain_onehot = nn.Embedding.from_pretrained(
            torch.eye(NUM_TERRAIN + 1)[..., 1:]
        )

        # self.entity_transformer = Transformer(
        #     transformer_num_layers=1,
        #     transformer_num_heads=2,
        #     transformer_key_size=entity_size // 2,
        #     transformer_value_size=entity_size // 2,
        #     transformer_model_size=entity_size,
        #     resblocks_num_before=1,
        #     resblocks_num_after=1,
        #     resblocks_hidden_size=entity_size // 2,
        # )
        self.to_vector = ToVector(
            entity_size, [vector_size], use_layer_norm=_USE_LAYER_NORM
        )

        self.turn_embedding = _layer_init(nn.Embedding(512, vector_size))

        boosts_size = NUM_BOOSTS + 12 * NUM_BOOSTS
        self.boosts_embedding = MLP(
            [2 * boosts_size, vector_size], use_layer_norm=_USE_LAYER_NORM
        )

        self.volatile_status_embedding = MLP(
            [2 * NUM_VOLATILE_STATUS, vector_size], use_layer_norm=_USE_LAYER_NORM
        )

        side_condition_size = NUM_SIDE_CONDITIONS + 2 + 3
        self.side_condition_embedding = MLP(
            [2 * side_condition_size, vector_size], use_layer_norm=_USE_LAYER_NORM
        )

        field_size = NUM_PSEUDOWEATHER + NUM_WEATHER + NUM_TERRAIN
        self.field_embedding = MLP(
            [field_size, vector_size], use_layer_norm=_USE_LAYER_NORM
        )

        self.action_hist = MLP(
            [4 * entity_size, vector_size, vector_size],
            use_layer_norm=_USE_LAYER_NORM,
        )

        self.torso_merge = VectorMerge([vector_size] * 6, vector_size)

        self.action_stage_embedding = _layer_init(nn.Embedding(3, vector_size))

        self.lstm = nn.LSTM(
            input_size=vector_size,
            hidden_size=vector_size,
            num_layers=3,
            batch_first=True,
        )

        self.action_type_mlp = MLP([vector_size, 2], use_layer_norm=_USE_LAYER_NORM)

        self.move_query_mlp = MLP(
            [vector_size, vector_size], use_layer_norm=_USE_LAYER_NORM
        )
        self.move_pointer = PointerLogits(
            vector_size,
            entity_size,
            key_size=entity_size,
            num_layers_keys=1,
            num_layers_query=1,
            use_layer_norm=_USE_LAYER_NORM,
        )

        self.switch_query_mlp = MLP(
            [vector_size, vector_size], use_layer_norm=_USE_LAYER_NORM
        )
        self.switch_pointer = PointerLogits(
            vector_size,
            entity_size,
            key_size=entity_size,
            num_layers_keys=1,
            num_layers_query=1,
            use_layer_norm=_USE_LAYER_NORM,
        )

        self.value = MLP(
            [vector_size, vector_size, vector_size, 1], use_layer_norm=_USE_LAYER_NORM
        )

    def encode_action_history(
        self, entity_embeddings: torch.Tensor, action_history: torch.Tensor
    ):
        T, B, H, *_ = entity_embeddings.shape

        action_hist_mask = action_history[..., 0] >= 0
        action_side = action_history[..., 0].clamp(min=0)
        action_user = action_history[..., 1].clamp(min=0)
        action_move = action_history[..., 2] + 2

        indices = F.one_hot(action_user + (action_side == 1) * 6, 18)
        action_user_embeddings = indices.float() @ entity_embeddings.flatten(-3, -2)

        action_embeddings = self.moves_onehot(action_move)
        action_embeddings = action_user_embeddings * torch.sigmoid(action_embeddings)
        action_embeddings = action_embeddings * action_hist_mask.unsqueeze(-1)

        action_hist_embedding = self.action_hist(action_embeddings.flatten(-2))
        action_hist_embedding = action_hist_embedding.view(T, B, H, -1)

        return action_hist_embedding

    def get_current_switch_embeddings(
        self, history_mask: torch.Tensor, entity_embeddings: torch.Tensor
    ):
        T, B, H = history_mask.shape
        my_switch_embeddings = entity_embeddings.flatten(0, 1)[:, :, 0]
        history_mask_onehot = (
            F.one_hot(history_mask.sum(-1) - 1, H)
            .flatten(0, 1)[..., None, None]
            .transpose(-3, -2)
            .detach()
        )

        current_ts_embeddings = (
            my_switch_embeddings.transpose(-3, -1) @ history_mask_onehot.float()
        ).transpose(-3, -1)

        return current_ts_embeddings.view(T, B, 6, -1)

    def encode_history(
        self, state_embeddings: torch.Tensor, seq_lens: torch.Tensor
    ) -> torch.Tensor:
        packed_input = pack_padded_sequence(
            state_embeddings,
            seq_lens.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (ht, _) = self.lstm(packed_input)
        return ht[-1]

    def get_action_mask_context(self, action_mask: torch.Tensor) -> torch.Tensor:
        action_type_select = torch.any(action_mask[..., :2], dim=-1)
        move_select = torch.any(action_mask[..., 2:6], dim=-1)
        switch_select = torch.any(action_mask[..., 6:], dim=-1)
        stage_token = action_type_select * 0 + move_select * 1 + switch_select * 2
        return self.action_stage_embedding(stage_token)

    def encode_teams(self, teams: torch.Tensor) -> torch.Tensor:
        T, B, H, *_ = teams.shape
        teamsp1 = teams + 1
        teamsp2 = teams + 2
        species_onehot = self.species_onehot(teamsp1[..., 0])
        item_onehot = self.item_onehot(teamsp1[..., 1])
        ability_onehot = self.ability_onehot(teamsp1[..., 2])
        # hp_value = teams[..., 3, None].float() / 1024
        hp_bucket = torch.sqrt(teams[..., 3]).to(torch.long)
        hp_onehot = self.hp_onehot(hp_bucket)
        active_onehot = self.active_onehot(teams[..., 4])
        fainted_onehot = self.fainted_onehot(teams[..., 5])
        status_onehot = self.status_onehot(teamsp1[..., 6])
        moveset_onehot = self.moves_onehot(teamsp2[..., -4:]).sum(-2)
        public_onehot = self.public_onehot(
            self.public_token[(None,) * 3].expand(T, B, H, 3, 6)
        )
        side_onehot = self.side_onehot(
            self.side_token[(None,) * 3].expand(T, B, H, 3, 6)
        )

        # encodings = torch.cat(
        #     (
        #         species_onehot,
        #         item_onehot,
        #         ability_onehot,
        #         hp_onehot,
        #         hp_value,
        #         active_onehot,
        #         fainted_onehot,
        #         status_onehot,
        #         moveset_onehot,
        #         public_onehot,
        #         side_onehot,
        #     ),
        #     dim=-1,
        # )
        encodings = (
            species_onehot
            + item_onehot
            + ability_onehot
            + hp_onehot
            # + hp_value
            + active_onehot
            + fainted_onehot
            + status_onehot
            + moveset_onehot
            + public_onehot
            + side_onehot
        )
        return self.units_mlp(encodings)

    def encode_side_conditions(self, side_conditions: torch.Tensor):
        other = side_conditions > 0
        spikes = self.spikes_onehot(side_conditions[..., 9])
        tspikes = self.tspikes_onehot(side_conditions[..., 13])
        encoding = torch.cat((other, spikes, tspikes), -1)
        return self.side_condition_embedding(encoding.flatten(-2))

    def encode_volatile_status(self, volatile_status: torch.Tensor) -> torch.Tensor:
        volatile_status_id = volatile_status[..., 0]
        volatile_status_level = volatile_status[..., 1]
        encoding = self.volatile_status_onehot(volatile_status_id + 1).sum(-2)
        return self.volatile_status_embedding(encoding.flatten(-2))

    def encode_boosts(self, boosts: torch.Tensor) -> torch.Tensor:
        boosts_onehot = self.boosts_onehot(boosts + 6)
        boosts_onehot = torch.cat((boosts_onehot[..., :6], boosts_onehot[..., 7:]), -1)
        boosts_scaled = torch.sign(boosts) * torch.sqrt(abs(boosts))
        encoding = torch.cat((boosts_onehot.flatten(-2), boosts_scaled), -1)
        return self.boosts_embedding(encoding.flatten(-2))

    def encode_field(self, field: torch.Tensor) -> torch.Tensor:
        field_id = field[..., 0]
        field_min_durr = field[..., 1]
        field_max_durr = field[..., 2]

        pseudoweathers = field_id[..., :3]
        pseudoweathers_onehot = self.pseudoweathers_onehot(pseudoweathers + 1).sum(-2)

        weather = field_id[..., 3]
        weather_onehot = self.weathers_onehot(weather + 1)

        terrain = field_id[..., 4]
        terrain_onehot = self.terrain_onehot(terrain + 1)

        encoding = torch.cat(
            (pseudoweathers_onehot, weather_onehot, terrain_onehot), -1
        )
        return self.field_embedding(encoding)

    def forward(
        self,
        turn: torch.Tensor,
        active_moveset: torch.Tensor,
        teams: torch.Tensor,
        side_conditions: torch.Tensor,
        volatile_status: torch.Tensor,
        boosts: torch.Tensor,
        field: torch.Tensor,
        legal: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
    ):
        T, B, H, *_ = teams.shape

        entity_embeddings = self.encode_teams(teams)

        # entity_embeddings_attn = self.entity_transformer(
        #     entity_embeddings.flatten(-3, -2),
        #     torch.ones_like(
        #         entity_embeddings.flatten(-3, -2)[..., 0].squeeze(-1), dtype=torch.bool
        #     ),
        # )
        # entity_embeddings = entity_embeddings_attn.view_as(entity_embeddings)
        entities_embedding = self.to_vector(entity_embeddings)

        side_conditions_encoding = self.encode_side_conditions(side_conditions)
        volatile_status_encoding = self.encode_volatile_status(volatile_status)
        boosts_encoding = self.encode_boosts(boosts)
        field_encoding = self.encode_field(field)

        action_hist_embedding = self.encode_action_history(entity_embeddings, history)

        state_embedding = self.torso_merge(
            entities_embedding,
            side_conditions_encoding,
            volatile_status_encoding,
            boosts_encoding,
            field_encoding,
            action_hist_embedding,
        )

        hist_mask = history_mask.sum(-1).flatten()
        state_embedding = self.encode_history(
            state_embedding.view(T * B, H, -1), hist_mask
        )
        state_embedding = state_embedding.view(T, B, -1)
        state_embedding = state_embedding + self.get_action_mask_context(legal)

        switch_embeddings = self.get_current_switch_embeddings(
            history_mask, entity_embeddings
        )
        move_embeddings = self.moves_onehot(active_moveset)

        action_type_logits = self.action_type_mlp(state_embedding)

        move_query = self.move_query_mlp(state_embedding).unsqueeze(-2)
        move_logits = self.move_pointer(move_query, move_embeddings).flatten(2)

        switch_query = self.switch_query_mlp(state_embedding).unsqueeze(-2)
        switch_logits = self.switch_pointer(switch_query, switch_embeddings).flatten(2)

        logits = torch.cat((action_type_logits, move_logits, switch_logits), dim=-1)
        policy = _legal_policy(logits, legal)
        log_policy = _legal_log_policy(logits, legal)

        value = self.value(state_embedding)

        return ModelOutput(
            logits=logits,
            policy=policy,
            log_policy=log_policy,
            value=value,
        )
