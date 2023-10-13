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
from pokesim.rl_utils import _legal_log_policy, _legal_policy

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


def ghostmax(x, dim=None):
    # subtract the max for stability
    x = x - x.max(dim=dim, keepdim=True).values
    # compute exponentials
    exp_x = torch.exp(x)
    # compute softmax values and add on in the denominator
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))


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


class RMSNorm(nn.Module):
    def __init__(self, d: int, p: float = -1.0, eps: float = 1e-8, bias: bool = False):
        super().__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x: torch.Tensor):
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


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

        self.gate_mlps = nn.ModuleList(
            [
                MLP([input_size, input_size], use_layer_norm)
                for input_size in input_sizes
            ]
        )
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
        for input, gate_mlp, out_lin in zip(inputs, self.gate_mlps, self.out_lins):
            feature = gate_mlp(input)
            output = out_lin(feature)
            gate.append(feature)
            outputs.append(output)
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
    def __init__(self, entity_size: int = 32, vector_size: int = 128):
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

        # self.units_lin1 = _layer_init(
        #     nn.Linear(
        #         self.species_onehot.weight.shape[-1]
        #         + self.item_onehot.weight.shape[-1]
        #         + self.ability_onehot.weight.shape[-1]
        #         + self.moves_onehot.weight.shape[-1]
        #         + self.hp_onehot.weight.shape[-1]
        #         + self.status_onehot.weight.shape[-1]
        #         + 3,
        #         entity_size,
        #     )
        # )

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

        boosts_size = NUM_BOOSTS + 12 * NUM_BOOSTS
        field_size = NUM_PSEUDOWEATHER + NUM_WEATHER + NUM_TERRAIN
        side_condition_size = NUM_SIDE_CONDITIONS + 2 + 3
        self.context_embedding = MLP(
            [
                2 * (boosts_size + side_condition_size + NUM_VOLATILE_STATUS)
                + TURN_ENC_SIZE
                + field_size,
                vector_size,
                vector_size,
            ],
            use_layer_norm=_USE_LAYER_NORM,
        )

        self.action_hist = MLP(
            [4 * 2 * entity_size, vector_size, vector_size],
            use_layer_norm=_USE_LAYER_NORM,
        )

        # self.state_merge = VectorMerge(
        #     [vector_size, vector_size, vector_size],
        #     vector_size,
        #     use_layer_norm=_USE_LAYER_NORM,
        # )
        self.state_merge = MLP(
            [vector_size, vector_size], use_layer_norm=_USE_LAYER_NORM
        )
        self.mask_lin = _layer_init(nn.Linear(12, vector_size))

        self.lstm = nn.LSTM(
            input_size=vector_size,
            hidden_size=vector_size,
            num_layers=3,
            batch_first=True,
        )

        self.action_type_resnet = ResNet(
            vector_size, num_resblocks=4, use_layer_norm=_USE_LAYER_NORM
        )
        self.action_type_mlp = MLP([vector_size, 2], use_layer_norm=_USE_LAYER_NORM)

        self.move_query_resnet = ResNet(
            vector_size, num_resblocks=1, use_layer_norm=_USE_LAYER_NORM
        )
        self.move_pointer = PointerLogits(
            vector_size,
            entity_size,
            key_size=entity_size,
            num_layers_keys=1,
            num_layers_query=1,
            use_layer_norm=_USE_LAYER_NORM,
        )

        self.switch_query_resnet = ResNet(
            vector_size, num_resblocks=1, use_layer_norm=_USE_LAYER_NORM
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

    def encode_history(self, state_embeddings: torch.Tensor, seq_lens: torch.Tensor):
        packed_input = pack_padded_sequence(
            state_embeddings,
            seq_lens.cpu().numpy(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (ht, _) = self.lstm(packed_input)
        return ht[-1]

    def encode_teams(self, teams: torch.Tensor):
        T, B, H, *_ = teams.shape
        teamsp1 = teams + 1
        teamsp2 = teams + 2
        species_onehot = self.species_onehot(teamsp1[..., 0])
        item_onehot = self.item_onehot(teamsp1[..., 1])
        ability_onehot = self.ability_onehot(teamsp1[..., 2])
        # hp_value = self.hp_value(teams[..., 3, None].float() / 1024)
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
        return torch.cat((other, spikes, tspikes), -1)

    def encode_side_conditions(self, side_conditions: torch.Tensor):
        other = side_conditions > 0
        spikes = self.spikes_onehot(side_conditions[..., 9])
        tspikes = self.tspikes_onehot(side_conditions[..., 13])
        return torch.cat((other, spikes, tspikes), -1)

    def encode_volatile_status(self, volatile_status: torch.Tensor) -> torch.Tensor:
        volatile_status_id = volatile_status[..., 0]
        volatile_status_level = volatile_status[..., 1]
        return self.volatile_status_onehot(volatile_status_id + 1).sum(-2)

    def encode_boosts(self, boosts: torch.Tensor):
        boosts_onehot = self.boosts_onehot(boosts + 6)
        boosts_onehot = torch.cat((boosts_onehot[..., :6], boosts_onehot[..., 7:]), -1)
        boosts_scaled = torch.sign(boosts) * torch.sqrt(abs(boosts))
        return torch.cat((boosts_onehot.flatten(-2), boosts_scaled), -1)

    def encode_field(self, field: torch.Tensor):
        field_id = field[..., 0]
        field_min_durr = field[..., 1]
        field_max_durr = field[..., 2]

        pseudoweathers = field_id[..., :3]
        pseudoweathers_onehot = self.pseudoweathers_onehot(pseudoweathers + 1).sum(-2)

        weather = field_id[..., 3]
        weather_onehot = self.weathers_onehot(weather + 1)

        terrain = field_id[..., 4]
        terrain_onehot = self.terrain_onehot(terrain + 1)

        return torch.cat((pseudoweathers_onehot, weather_onehot, terrain_onehot), -1)

    def _threshold(
        self, policy: torch.Tensor, mask: torch.Tensor, threshold: float = 0.05
    ) -> torch.Tensor:
        """Remove from the support the actions 'a' where policy(a) < threshold."""
        mask = mask * (
            # Values over the threshold.
            (policy >= threshold)
            +
            # Degenerate case is when policy is less than threshold *everywhere*.
            # In that case we just keep the policy as-is.
            (torch.max(policy, dim=-1, keepdim=True).values < threshold)
        )
        return mask * policy / torch.sum(mask * policy, dim=-1, keepdim=True)

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
        context_encoding = torch.cat(
            (
                side_conditions_encoding.flatten(-2),
                volatile_status_encoding.flatten(-2),
                boosts_encoding.flatten(-2),
                field_encoding,
                turn,
            ),
            dim=-1,
        )
        context_embedding = self.context_embedding(context_encoding)

        user = history[..., 0].clamp(min=0)
        user = torch.where(user >= 12, user - 6, user)
        action = history[..., 1] + 1

        action_move_embeddings = self.moves_onehot(action + 1)
        action_hist_mask = history[..., 0] >= 0

        user_index = torch.arange(T * B * H, device=user.device).unsqueeze(-1)
        user_index *= entity_embeddings.shape[-2]
        user_index = user_index + user.flatten(0, -2)
        user_embeddings = torch.embedding(entity_embeddings.flatten(0, -2), user_index)
        user_embeddings = user_embeddings.view(T, B, H, 4, -1)

        action_hist_embeddings = torch.cat(
            (action_move_embeddings, user_embeddings), dim=-1
        ) * action_hist_mask.unsqueeze(-1)
        action_hist_embedding = self.action_hist(action_hist_embeddings.flatten(-2))
        action_hist_embedding = action_hist_embedding.view(T, B, H, -1)

        state_embedding = self.state_merge(
            entities_embedding + context_embedding + action_hist_embedding
        )

        hist_mask = history_mask.sum(-1).flatten()
        state_embedding = self.encode_history(
            state_embedding.view(T * B, H, -1), hist_mask
        )
        state_embedding = state_embedding.view(T, B, -1)
        state_embedding = state_embedding + self.mask_lin(legal.float())

        current_ts_index = torch.arange(T * B * H, device=user.device)
        current_ts_index = (
            F.one_hot(hist_mask - 1, H) * current_ts_index.view(T * B, H)
        ).sum(-1)
        current_ts_entity_embeddings = torch.embedding(
            entity_embeddings.view(T * B * H, 3, -1)[:, 0], current_ts_index
        )
        switch_embeddings = current_ts_entity_embeddings.view(T, B, 6, -1)

        move_embeddings = self.moves_onehot(active_moveset)

        action_type_query = self.action_type_resnet(state_embedding)
        action_type_logits = self.action_type_mlp(action_type_query)

        move_query = self.move_query_resnet(state_embedding).unsqueeze(-2)
        move_logits = self.move_pointer(move_query, move_embeddings).flatten(2)

        switch_query = self.switch_query_resnet(state_embedding).unsqueeze(-2)
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
