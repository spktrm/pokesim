import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence

from enum import Enum
from typing import Dict, Optional, Sequence, Tuple

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
        use_layer_norm: bool = True,
    ):
        super().__init__()

        self.input_size = input_size
        self.net = MLP([input_size, input_size], use_layer_norm=use_layer_norm)
        self.gate = MLP([input_size, 1], use_layer_norm=use_layer_norm)
        out_layers = [
            nn.ReLU(),
            _layer_init(nn.Linear(2 * input_size, 2 * input_size)),
        ]
        if use_layer_norm:
            out_layers.insert(0, nn.LayerNorm(2 * input_size))
        self.out = nn.Sequential(*out_layers)

    def forward(
        self, entity_embeddings: torch.Tensor, active_token: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        entity_embeddings = self.net(entity_embeddings)
        active_weight = active_token.unsqueeze(-2)
        active_embedding = active_weight.float() @ entity_embeddings
        reserve_weight = torch.where(
            active_weight == 1, -1e9, self.gate(entity_embeddings).transpose(-2, -1)
        ).softmax(-1)
        reserve_embeddings = reserve_weight @ entity_embeddings
        entity_embeddings = torch.cat(
            (active_embedding, reserve_embeddings), dim=-1
        ).flatten(-2)
        return torch.chunk(self.out(entity_embeddings), 3, -2)


class GatingType(Enum):
    NONE = 0
    GLOBAL = 1
    POINTWISE = 2


class VectorMerge(nn.Module):
    def __init__(
        self,
        input_sizes: Dict[str, Optional[int]],
        output_size: int,
        gating_type: GatingType = GatingType.NONE,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        if not input_sizes:
            raise ValueError("input_sizes cannot be empty")

        self.input_sizes = input_sizes
        self.output_size = output_size
        self.gating_type = gating_type
        self.use_layer_norm = use_layer_norm

        self.linear_layers = nn.ModuleDict(
            {
                name: _layer_init(
                    nn.Linear(size if size is not None else 1, output_size)
                )
                for name, size in input_sizes.items()
            }
        )

        if gating_type != GatingType.NONE:
            self.gate_layers = nn.ModuleDict(
                {
                    name: _layer_init(
                        nn.Linear(
                            output_size,
                            output_size if gating_type == GatingType.POINTWISE else 1,
                        )
                    )
                    for name in input_sizes.keys()
                }
            )

        self.layer_norms = (
            nn.ModuleDict(
                {name: nn.LayerNorm(output_size) for name in input_sizes.keys()}
            )
            if use_layer_norm
            else None
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        encoded = []
        gates = []

        for name in self.input_sizes.keys():
            x = inputs[name]
            if self.input_sizes[name] is None:
                x = x.unsqueeze(-1)

            if self.use_layer_norm:
                x = F.relu(self.layer_norms[name](self.linear_layers[name](x)))
            else:
                x = F.relu(self.linear_layers[name](x))

            encoded.append(x)

            if self.gating_type != GatingType.NONE:
                gates.append(self.gate_layers[name](x))

        if self.gating_type == GatingType.NONE:
            output = sum(encoded)
        else:
            if self.gating_type == GatingType.GLOBAL:
                gate = torch.sigmoid(sum(gates))
                output = sum(g * d for g, d in zip(gates, encoded))
            elif self.gating_type == GatingType.POINTWISE:
                gate = F.softmax(torch.stack(gates, dim=0), dim=0)
                output = (torch.stack(encoded) * gate).sum(0)
            else:
                raise ValueError(f"Gating type {self.gating_type} is not supported")

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


class SimSiam(nn.Module):
    def __init__(self, vector_size: int) -> None:
        super().__init__()

        self.proj = MLP(
            [vector_size, vector_size, vector_size], use_layer_norm=_USE_LAYER_NORM
        )
        self.proj_head = MLP([vector_size, vector_size], use_layer_norm=_USE_LAYER_NORM)

    def forward(self, x_pred: torch.Tensor, x: torch.Tensor):
        x_pred = self.proj_head(self.proj(x_pred))
        x_pred = F.normalize(x_pred, p=2.0, dim=-1, eps=1e-5)
        x = self.proj(x).detach()
        x = F.normalize(x, p=2.0, dim=-1, eps=1e-5)
        return -(x * x_pred).sum(-1)


class Model(nn.Module):
    def __init__(self, entity_size: int = 256, vector_size: int = 512):
        super().__init__()

        self.entity_size = entity_size
        self.vector_size = vector_size

        side_token = torch.zeros(18, dtype=torch.long).view(-1, 6)
        side_token[-1] = 1
        self.register_buffer("side_token", side_token)

        self.species_onehot = _layer_init(nn.Embedding(NUM_SPECIES + 1, entity_size))
        self.item_onehot = _layer_init(nn.Embedding(NUM_ITEMS + 1, entity_size))
        self.ability_onehot = _layer_init(nn.Embedding(NUM_ABILITIES + 1, entity_size))
        self.moves_onehot = _layer_init(nn.Embedding(NUM_MOVES + 2, entity_size))
        self.hp_onehot = _layer_init(nn.Embedding(NUM_HP_BUCKETS + 1, entity_size))
        self.status_onehot = _layer_init(nn.Embedding(NUM_STATUS + 1, entity_size))
        self.active_onehot = _layer_init(nn.Embedding(2, entity_size))
        self.fainted_onehot = _layer_init(nn.Embedding(2, entity_size))
        self.side_onehot = _layer_init(nn.Embedding(2, entity_size))
        self.public_onehot = _layer_init(nn.Embedding(2, entity_size))

        self.units_mlp = MLP([entity_size, entity_size], use_layer_norm=_USE_LAYER_NORM)
        self.moves_mlp = MLP([entity_size, entity_size], use_layer_norm=_USE_LAYER_NORM)

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
        self.to_vector = ToVector(entity_size, use_layer_norm=_USE_LAYER_NORM)

        self.turn_embedding = _layer_init(nn.Embedding(64, vector_size))

        boosts_size = NUM_BOOSTS + 12 * NUM_BOOSTS
        self.boosts_embedding = MLP(
            [boosts_size, vector_size], use_layer_norm=_USE_LAYER_NORM
        )

        self.volatile_status_embedding = MLP(
            [NUM_VOLATILE_STATUS, vector_size], use_layer_norm=_USE_LAYER_NORM
        )

        side_condition_size = NUM_SIDE_CONDITIONS + 2 + 3
        self.side_condition_embedding = MLP(
            [side_condition_size, vector_size], use_layer_norm=_USE_LAYER_NORM
        )

        field_size = NUM_PSEUDOWEATHER + NUM_WEATHER + NUM_TERRAIN
        self.field_embedding = MLP(
            [field_size, vector_size], use_layer_norm=_USE_LAYER_NORM
        )

        self.action_rnn = nn.GRU(
            input_size=entity_size,
            hidden_size=vector_size,
            num_layers=1,
            batch_first=True,
        )

        self.side_merge = VectorMerge(
            {
                "side": 2 * entity_size,
                "side_conditions": vector_size,
                "volatile_status": vector_size,
                "boosts": vector_size,
            },
            vector_size,
            gating_type=GatingType.NONE,
            use_layer_norm=_USE_LAYER_NORM,
        )

        self.torso_merge = VectorMerge(
            {
                "myside": vector_size,
                "oppside": vector_size,
                "field": vector_size,
                "action": vector_size,
                "private": 2 * entity_size,
                "turn": vector_size,
            },
            vector_size,
            gating_type=GatingType.NONE,
            use_layer_norm=_USE_LAYER_NORM,
        )

        self.action_stage_embedding = _layer_init(nn.Embedding(3, vector_size))

        self.state_rnn = nn.GRU(
            input_size=vector_size,
            hidden_size=vector_size,
            num_layers=1,
            batch_first=True,
        )

        self.action_type_mlp = MLP(
            [vector_size, vector_size, 2], use_layer_norm=_USE_LAYER_NORM
        )

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

        self.value_resnet = ResNet(vector_size, use_layer_norm=_USE_LAYER_NORM)
        self.value_mlp = MLP([vector_size, 1], use_layer_norm=_USE_LAYER_NORM)

    def embed_action_history(
        self, entity_embeddings: torch.Tensor, action_history: torch.Tensor
    ):
        T, B, H, *_ = entity_embeddings.shape

        action_hist_mask = action_history[..., 0] >= 0
        action_hist_len = action_hist_mask.sum(-1).flatten().clamp(min=1)
        action_side = action_history[..., 0].clamp(min=0)
        action_user = action_history[..., 1].clamp(min=0)
        action_move = action_history[..., 2] + 2

        indices = F.one_hot(action_user + (action_side == 1) * 6, 18)
        action_user_embeddings = indices.float() @ entity_embeddings.flatten(-3, -2)

        action_embeddings = self.moves_onehot(action_move)
        action_embeddings = action_embeddings * torch.sigmoid(action_user_embeddings)
        action_embeddings = (
            action_embeddings * action_hist_mask.unsqueeze(-1)
        ).flatten(0, -3)

        packed_input = pack_padded_sequence(
            action_embeddings,
            action_hist_len.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, ht = self.action_rnn(packed_input)
        return ht[-1].view(T, B, H, -1)

    def encode_history(
        self, state_embeddings: torch.Tensor, history_mask: torch.Tensor
    ) -> torch.Tensor:
        T, B, H, *_ = state_embeddings.shape
        history_mask_onehot = F.one_hot(history_mask.sum(-1) - 1, H).float()
        output, _ = self.state_rnn(state_embeddings.flatten(0, 1))
        output = output.view(T, B, H, -1)
        final_state = history_mask_onehot.unsqueeze(-2) @ output
        final_state = final_state.view(T, B, -1)
        return output, final_state

    def get_current_entity_embeddings(
        self,
        history_mask: torch.Tensor,
        entity_embeddings: torch.Tensor,
        # active_token: torch.Tensor,
        # ) -> Tuple[torch.Tensor, torch.Tensor]:
    ) -> torch.Tensor:
        T, B, H = history_mask.shape

        my_switch_embeddings = entity_embeddings[:, :, :, 0].transpose(2, -1)
        history_mask_onehot = (
            F.one_hot(history_mask.sum(-1) - 1, H)[..., None, None]
            .transpose(2, -2)
            .detach()
            .float()
        )

        current_ts_embeddings = my_switch_embeddings @ history_mask_onehot
        current_ts_embeddings = current_ts_embeddings.transpose(-3, -1)

        # current_active_token = (
        #     active_token[:, :, :, 0].transpose(2, -1).float() @ history_mask_onehot
        # ).transpose(-2, -1)

        current_ts_embeddings = current_ts_embeddings.view(T, B, 6, -1)
        # current_active_embedding = current_active_token @ current_ts_embeddings
        # current_active_embedding = current_active_embedding.view(T, B, 1, -1)

        return current_ts_embeddings  # , current_active_embedding

    def get_action_mask_context(self, action_mask: torch.Tensor) -> torch.Tensor:
        action_type_select = torch.any(action_mask[..., :2], dim=-1)
        move_select = torch.any(action_mask[..., 2:6], dim=-1)
        switch_select = torch.any(action_mask[..., 6:], dim=-1)
        stage_token = action_type_select * 0 + move_select * 1 + switch_select * 2
        return self.action_stage_embedding(stage_token)

    def embed_teams(self, teams: torch.Tensor) -> torch.Tensor:
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
        # public_onehot = self.public_onehot(
        #     self.public_token[(None,) * 3].expand(T, B, H, 3, 6)
        # )
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
            # + public_onehot
            + side_onehot
        )
        return self.units_mlp(encodings)

    def embed_side_conditions(
        self, side_conditions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        other = side_conditions > 0
        spikes = self.spikes_onehot(side_conditions[..., 9])
        tspikes = self.tspikes_onehot(side_conditions[..., 13])
        encoding = torch.cat((other, spikes, tspikes), -1)
        return torch.chunk(self.side_condition_embedding(encoding), 2, -2)

    def embed_volatile_status(
        self, volatile_status: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        volatile_status_id = volatile_status[..., 0]
        volatile_status_level = volatile_status[..., 1]
        encoding = self.volatile_status_onehot(volatile_status_id + 1).sum(-2)
        return torch.chunk(self.volatile_status_embedding(encoding), 2, -2)

    def embed_boosts(self, boosts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        boosts_onehot = self.boosts_onehot(boosts + 6)
        boosts_onehot = torch.cat((boosts_onehot[..., :6], boosts_onehot[..., 7:]), -1)
        boosts_scaled = torch.sign(boosts) * torch.sqrt(abs(boosts))
        encoding = torch.cat((boosts_onehot.flatten(-2), boosts_scaled), -1)
        return torch.chunk(self.boosts_embedding(encoding), 2, -2)

    def softmax_passthrough(self, tensor: torch.Tensor, buckets: int = 16):
        og_shape = tensor.shape
        flat_tensor = tensor.reshape(-1, buckets)
        gsm = F.gumbel_softmax(flat_tensor)
        return gsm.view(*og_shape)

    def embed_field(self, field: torch.Tensor) -> torch.Tensor:
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
        entity_embeddings = self.embed_teams(teams)

        # entity_embeddings_attn = self.entity_transformer(
        #     entity_embeddings.flatten(-3, -2),
        #     torch.ones_like(
        #         entity_embeddings.flatten(-3, -2)[..., 0].squeeze(-1), dtype=torch.bool
        #     ),
        # )
        # entity_embeddings = entity_embeddings_attn.view_as(entity_embeddings)

        active_token = teams[..., 4]
        (
            my_private_side_embedding,
            my_public_side_embedding,
            opp_side_embedding,
        ) = self.to_vector(entity_embeddings, active_token)

        private_information = my_private_side_embedding - my_public_side_embedding

        (
            my_side_conditions_embedding,
            opp_side_conditions_embedding,
        ) = self.embed_side_conditions(side_conditions)
        (
            my_volatile_status_embedding,
            opp_volatile_status_embedding,
        ) = self.embed_volatile_status(volatile_status)
        my_boosts_embedding, opp_boosts_embedding = self.embed_boosts(boosts)

        my_side_embedding = self.side_merge(
            {
                "side": my_private_side_embedding.squeeze(-2),
                "side_conditions": my_side_conditions_embedding.squeeze(-2),
                "volatile_status": my_volatile_status_embedding.squeeze(-2),
                "boosts": my_boosts_embedding.squeeze(-2),
            }
        )
        opp_side_embedding = self.side_merge(
            {
                "side": opp_side_embedding.squeeze(-2),
                "side_conditions": opp_side_conditions_embedding.squeeze(-2),
                "volatile_status": opp_volatile_status_embedding.squeeze(-2),
                "boosts": opp_boosts_embedding.squeeze(-2),
            }
        )

        field_embedding = self.embed_field(field)

        prev_entity_embeddings = torch.cat(
            (entity_embeddings[:, :, :1], entity_embeddings[:, :, :-1]), dim=2
        )
        action_hist_embedding = self.embed_action_history(
            prev_entity_embeddings, history
        )
        turn_embedding = self.turn_embedding(
            turn.clamp(min=0, max=self.turn_embedding.weight.shape[0] - 1)
        )

        step_embeddings = self.torso_merge(
            {
                "myside": my_side_embedding,
                "oppside": opp_side_embedding,
                "field": field_embedding,
                "action": action_hist_embedding,
                "private": private_information.squeeze(-2),
                "turn": turn_embedding,
            }
        )

        step_embeddings, state_embedding = self.encode_history(
            step_embeddings, history_mask
        )

        state_embedding_w_context = state_embedding + self.get_action_mask_context(
            legal
        )
        # state_embedding_w_context = self.softmax_passthrough(state_embedding_w_context)

        active_token = teams[..., 4]
        switch_embeddings = self.get_current_entity_embeddings(
            history_mask, entity_embeddings  # , active_token
        )
        # switch_embeddings = self.softmax_passthrough(switch_embeddings)

        move_embeddings = self.moves_onehot(active_moveset)
        move_embeddings = self.moves_mlp(move_embeddings)
        # move_embeddings = self.softmax_passthrough(move_embeddings)

        action_type_logits = self.action_type_mlp(state_embedding_w_context)

        move_query = self.move_query_mlp(state_embedding_w_context).unsqueeze(-2)
        move_logits = self.move_pointer(move_query, move_embeddings).flatten(2)

        switch_query = self.switch_query_mlp(state_embedding_w_context).unsqueeze(-2)
        switch_logits = self.switch_pointer(switch_query, switch_embeddings).flatten(2)

        logits = torch.cat((action_type_logits, move_logits, switch_logits), dim=-1)

        policy = _legal_policy(logits, legal)
        log_policy = _legal_log_policy(logits, legal)

        value = self.value_mlp(self.value_resnet(state_embedding_w_context))

        return ModelOutput(
            logits=logits,
            policy=policy,
            log_policy=log_policy,
            value=value,
        )
