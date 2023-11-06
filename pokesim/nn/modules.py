import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from enum import Enum
from typing import Dict, Optional, Sequence, Tuple

from pokesim.nn.helpers import _layer_init

_USE_LAYER_NORM = False


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
        output_size: int,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.qk = MLP([input_size, 2 * input_size], use_layer_norm=use_layer_norm)
        self.denom = 1 / (input_size**0.5)

        self.v = MLP([input_size, output_size], use_layer_norm=use_layer_norm)
        self.out = MLP([output_size, output_size], use_layer_norm=use_layer_norm)

    def forward(
        self, entity_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        qk = self.qk(entity_embeddings)
        queries, keys = torch.chunk(qk, 2, -1)
        values = self.v(entity_embeddings)
        attention_logits = queries @ keys.transpose(-2, -1)
        attention_weights = (attention_logits * self.denom).softmax(-1)
        entity_embeddings = attention_weights @ values
        return self.out(entity_embeddings.mean(-2)), entity_embeddings


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


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = _layer_init(nn.Embedding(num_embeddings, embedding_dim))
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(-1, self.embedding_dim)
        d = (
            torch.sum(x_flat**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight**2, dim=1)
            - 2 * torch.matmul(x_flat, self.embeddings.weight.t())
        )
        min_encoding_indices = torch.argmin(d, dim=1)
        x_q = self.embeddings(min_encoding_indices)
        x_q = x_q.view_as(x)
        x_q = x + (x_q - x).detach()

        loss = ((x_q.detach() - x) ** 2) + self.beta * ((x_q - x.detach()) ** 2)
        loss = loss.mean(-1).flatten(3).mean(-1)

        return x_q, loss
