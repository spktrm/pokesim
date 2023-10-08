import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence

from typing import Sequence
from pokesim.data import (
    NUM_BOOSTS,
    NUM_MOVES,
    NUM_PSEUDOWEATHER,
    NUM_SIDE_CONDITIONS,
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
            std = math.sqrt(1 / n)
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
        x = self.net(x)
        x = self.out(x.flatten(-3, -2).max(-2).values)
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

        self.action_embeddings = _layer_init(nn.Embedding(NUM_MOVES + 1, entity_size))

        self.side_embedding = nn.Embedding(2, embedding_dim=entity_size)
        self.public_embedding = nn.Embedding(2, embedding_dim=entity_size)

        self.units_lin1 = _layer_init(nn.Linear(3289, entity_size))
        # self.units_lin2 = _layer_init(nn.Linear(entity_size, entity_size))

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
        self.to_vector = ToVector(entity_size, [vector_size])

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
        self.state_merge = MLP([vector_size, vector_size])
        self.mask_lin = _layer_init(nn.Linear(12, vector_size))

        self.lstm = nn.LSTM(
            input_size=vector_size,
            hidden_size=vector_size,
            num_layers=1,
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

        side_token = torch.zeros_like(teams[..., 0], dtype=torch.long)
        side_token[..., 2:, :] = 1

        public_token = torch.zeros_like(teams[..., 0], dtype=torch.long)
        public_token[..., 1:, :] = 1

        entity_embeddings = (
            self.units_lin1(teams)
            + self.public_embedding(public_token)
            + self.side_embedding(side_token)
        )
        # entity_embeddings = self.encode_units(entity_embeddings, public_token)

        # entity_embeddings_attn = self.entity_transformer(
        #     entity_embeddings.flatten(-3, -2),
        #     torch.ones_like(
        #         entity_embeddings.flatten(-3, -2)[..., 0].squeeze(-1), dtype=torch.bool
        #     ),
        # )
        # entity_embeddings = entity_embeddings_attn.view_as(entity_embeddings)
        entities_embedding = self.to_vector(entity_embeddings)

        context_embedding = self.context_embedding(
            torch.cat((side_conditions, volatile_status, boosts, field, turn), dim=-1)
        )

        user = history[..., 0].clamp(min=0)
        user = torch.where(user >= 12, user - 6, user)
        action = history[..., 1] + 1

        action_move_embeddings = self.action_embeddings(action)
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

        move_embeddings = self.action_embeddings(active_moveset)

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
