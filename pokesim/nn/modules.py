import math
from telnetlib import Telnet

import torch
import torch.nn as nn
import torch.nn.functional as F

from enum import Enum
from typing import Dict, Optional, Sequence, Tuple
from functools import partial

from pokesim.nn.helpers import _layer_init


class Resblock(nn.Module):
    """Fully connected residual block."""

    def __init__(
        self,
        input_size: int,
        num_layers: int = 2,
        hidden_size: Optional[int] = None,
        use_layer_norm: bool = True,
        affine_layer_norm: bool = False,
    ):
        """Initializes VectorResblock module.

        Args:
          num_layers: Number of layers in the residual block.
          hidden_size: Size of the activation vector in the residual block.
          use_layer_norm: Whether to use layer normalization.
          name: The name of this component.
        """
        super().__init__()
        self._input_size = input_size
        self._num_layers = num_layers
        self._hidden_size = hidden_size or input_size
        self._use_layer_norm = use_layer_norm

        # Create layers
        self.layers = nn.ModuleList()

        if num_layers < 1:
            raise ValueError("Must have at least 1 Layer")

        layer_sizes = [input_size, input_size]
        for _ in range(num_layers - 1):
            layer_sizes.insert(1, self._hidden_size)

        layer_init = partial(_layer_init, std=0.005, bias_value=0)

        for input_size, output_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            if use_layer_norm:
                self.layers.append(
                    nn.LayerNorm(input_size, elementwise_affine=affine_layer_norm)
                )
            self.layers.append(nn.ReLU())
            self.layers.append(layer_init(nn.Linear(input_size, output_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        for layer in self.layers:
            x = layer(x)
        return x + shortcut


class Resnet(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_resblocks: int,
        use_layer_norm: bool = True,
        affine_layer_norm: bool = False,
    ):
        super().__init__()
        self.resblocks = nn.ModuleList()
        for _ in range(num_resblocks):
            resblock = Resblock(
                input_size=input_size,
                use_layer_norm=use_layer_norm,
                affine_layer_norm=affine_layer_norm,
            )
            self.resblocks.append(resblock)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resblock in self.resblocks:
            x = resblock(x)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        layer_sizes: Sequence[int] = None,
        use_layer_norm: bool = True,
        affine_layer_norm: bool = False,
    ):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.layers = nn.ModuleList()

        for input_size, output_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            if use_layer_norm:
                self.layers.append(
                    nn.LayerNorm(input_size, elementwise_affine=affine_layer_norm)
                )
            self.layers.append(nn.ReLU())
            self.layers.append(_layer_init(nn.Linear(input_size, output_size)))

    @property
    def in_features(self):
        return self.layer_sizes[0]

    @property
    def out_features(self):
        return self.layer_sizes[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x.softmax(dim=dim)


def softmax_one(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # subtract the max for stability
    x = x - x.max(dim=dim, keepdim=True).values
    # compute exponentials
    exp_x = torch.exp(x)
    # compute softmax values and add on in the denominator
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        key_size: int,
        query_size: int = None,
        value_size: int = None,
        with_bias: bool = True,
        model_size: int = None,
        d_k: int = None,
        d_q: int = None,
        d_v: int = None,
    ):
        super().__init__()
        self._key_size = key_size
        self._query_size = query_size or key_size
        self._value_size = value_size or key_size
        self._num_heads = num_heads
        self._model_size = model_size or key_size * num_heads
        self._denom = 1 / math.sqrt(key_size)

        self.query_lin = _layer_init(
            nn.Linear(
                d_q or self._model_size,
                self._num_heads * self._key_size,
                bias=with_bias,
            )
        )
        self.key_lin = _layer_init(
            nn.Linear(
                d_k or self._model_size,
                self._num_heads * self._key_size,
                bias=with_bias,
            )
        )
        self.value_lin = _layer_init(
            nn.Linear(
                d_v or self._model_size,
                self._num_heads * self._value_size,
                bias=with_bias,
            )
        )
        self.final_proj = _layer_init(
            nn.Linear(
                self._num_heads * self._value_size,
                self._model_size,
            )
        )

    def _linear_projection(self, x: torch.Tensor, mod: nn.Module) -> torch.Tensor:
        y = mod(x)
        *leading_dims, _ = x.shape
        return y.reshape((*leading_dims, self._num_heads, -1))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        *leading_dims, sequence_length, _ = query.shape

        query_heads = self._linear_projection(query, self.query_lin)
        key_heads = self._linear_projection(key, self.key_lin)
        value_heads = self._linear_projection(value, self.value_lin)

        attn_logits = torch.einsum("...thd,...Thd->...htT", query_heads, key_heads)
        attn_logits = attn_logits * self._denom
        if mask is not None:
            if mask.ndim != attn_logits.ndim:
                raise ValueError(
                    f"Mask dimensionality {mask.ndim} must match logits dimensionality "
                    f"{attn_logits.ndim}."
                )
            attn_logits = torch.where(mask, attn_logits, -1e30)
        attn_weights = softmax(attn_logits, -1)

        attn = torch.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
        attn = torch.reshape(attn, (*leading_dims, sequence_length, -1))

        return self.final_proj(attn)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        units_stream_size: int,
        transformer_num_layers: int,
        transformer_num_heads: int,
        transformer_key_size: int,
        transformer_value_size: int,
        resblocks_num_before: int,
        resblocks_num_after: int,
        resblocks_hidden_size: Optional[int] = None,
        use_layer_norm: bool = True,
        affine_layer_norm: bool = True,
    ):
        super().__init__()

        self._units_stream_size = units_stream_size
        self._transformer_num_layers = transformer_num_layers
        self._transformer_num_heads = transformer_num_heads
        self._transformer_key_size = transformer_key_size
        self._transformer_value_size = transformer_value_size
        self._resblocks_num_before = resblocks_num_before
        self._resblocks_num_after = resblocks_num_after
        self._resblocks_hidden_size = resblocks_hidden_size
        self._use_layer_norm = use_layer_norm

        # Define the PyTorch modules here
        self.resblocks_before = nn.ModuleList(
            [
                Resblock(
                    input_size=units_stream_size,
                    hidden_size=self._resblocks_hidden_size,
                    use_layer_norm=self._use_layer_norm,
                    affine_layer_norm=affine_layer_norm,
                )
                for _ in range(self._resblocks_num_before)
            ]
        )
        if self._use_layer_norm:
            self._layernorms = nn.ModuleList(
                [
                    nn.LayerNorm(
                        self._units_stream_size, elementwise_affine=affine_layer_norm
                    )
                    for _ in range(self._transformer_num_layers)
                ]
            )
        self.transformer_layers = nn.ModuleList(
            [
                MultiHeadAttention(
                    num_heads=self._transformer_num_heads,
                    model_size=self._units_stream_size,
                    key_size=self._transformer_key_size,
                    value_size=self._transformer_value_size,
                )
                for _ in range(self._transformer_num_layers)
            ]
        )
        self.resblocks_after = nn.ModuleList(
            [
                Resblock(
                    input_size=units_stream_size,
                    hidden_size=self._resblocks_hidden_size,
                    use_layer_norm=self._use_layer_norm,
                    affine_layer_norm=affine_layer_norm,
                )
                for _ in range(self._resblocks_num_after)
            ]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits_mask = mask[..., None, None, :]
        for resblock in self.resblocks_before:
            x = resblock(x)
        for layer_index, layer in enumerate(self.transformer_layers):
            x1 = x
            if self._use_layer_norm:
                x1 = self._layernorms[layer_index](x1)
            x1 = F.relu(x1)
            x1 = layer(query=x1, key=x1, value=x1, mask=logits_mask)
            x1 = torch.where(mask[..., None], x1, 0)
            x = x + x1
        for resblock in self.resblocks_after:
            x = resblock(x)
        x = torch.where(mask[..., None], x, 0)
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        units_stream_size: int,
        transformer_num_layers: int,
        transformer_num_heads: int,
        transformer_key_size: int,
        transformer_value_size: int,
        resblocks_num_before: int,
        resblocks_num_after: int,
        resblocks_hidden_size: Optional[int] = None,
        use_layer_norm: bool = True,
        affine_layer_norm: bool = True,
    ):
        super().__init__()

        self._units_stream_size = units_stream_size
        self._transformer_num_layers = transformer_num_layers
        self._transformer_num_heads = transformer_num_heads
        self._transformer_key_size = transformer_key_size
        self._transformer_value_size = transformer_value_size
        self._resblocks_num_before = resblocks_num_before
        self._resblocks_num_after = resblocks_num_after
        self._resblocks_hidden_size = resblocks_hidden_size
        self._use_layer_norm = use_layer_norm

        # Define the PyTorch modules here
        self.resblocks_before = nn.ModuleList(
            [
                Resblock(
                    input_size=units_stream_size,
                    hidden_size=self._resblocks_hidden_size,
                    use_layer_norm=self._use_layer_norm,
                    affine_layer_norm=affine_layer_norm,
                )
                for _ in range(self._resblocks_num_before)
            ]
        )
        if self._use_layer_norm:
            self._layernorms1 = nn.ModuleList(
                [
                    nn.LayerNorm(
                        self._units_stream_size, elementwise_affine=affine_layer_norm
                    )
                    for _ in range(self._transformer_num_layers)
                ]
            )
            self._layernorms2 = nn.ModuleList(
                [
                    nn.LayerNorm(
                        self._units_stream_size, elementwise_affine=affine_layer_norm
                    )
                    for _ in range(self._transformer_num_layers)
                ]
            )
        self.self_attn_layers = nn.ModuleList(
            [
                MultiHeadAttention(
                    num_heads=self._transformer_num_heads,
                    model_size=self._units_stream_size,
                    key_size=self._transformer_key_size,
                    value_size=self._transformer_value_size,
                )
                for _ in range(self._transformer_num_layers)
            ]
        )
        self.cross_attn_layers = nn.ModuleList(
            [
                MultiHeadAttention(
                    num_heads=self._transformer_num_heads,
                    model_size=self._units_stream_size,
                    key_size=self._transformer_key_size,
                    value_size=self._transformer_value_size,
                )
                for _ in range(self._transformer_num_layers)
            ]
        )
        self.resblocks_after = nn.ModuleList(
            [
                Resblock(
                    input_size=units_stream_size,
                    hidden_size=self._resblocks_hidden_size,
                    use_layer_norm=self._use_layer_norm,
                    affine_layer_norm=affine_layer_norm,
                )
                for _ in range(self._resblocks_num_after)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask_x: torch.Tensor,
        mask_y: torch.Tensor,
    ) -> torch.Tensor:
        logits_mask = mask_y[..., None, None, :]
        for resblock in self.resblocks_before:
            y = resblock(y)
        for layer_index, layer in enumerate(self.self_attn_layers):
            y1 = y
            if self._use_layer_norm:
                y1 = self._layernorms1[layer_index](y1)
            y1 = F.relu(y1)
            y1 = layer(query=y1, key=y1, value=y1, mask=logits_mask)
            y1 = torch.where(mask_y[..., None], y1, 0)
            y = y + y1

        logits_mask = mask_x.unsqueeze(-1) * mask_y.unsqueeze(-2)
        logits_mask = logits_mask[..., None, :, :]

        for layer_index, layer in enumerate(self.cross_attn_layers):
            x1 = x
            if self._use_layer_norm:
                x1 = self._layernorms2[layer_index](x1)
            x1 = F.relu(x1)
            x1 = layer(query=x1, key=y, value=y, mask=logits_mask)
            x1 = torch.where(mask_x[..., None], x1, 0)
            x = x + x1
        for resblock in self.resblocks_after:
            x = resblock(x)
        x = torch.where(mask_x[..., None], x, 0)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        units_stream_size: int,
        transformer_num_layers: int,
        transformer_num_heads: int,
        transformer_key_size: int,
        transformer_value_size: int,
        resblocks_num_before: int,
        resblocks_num_after: int,
        resblocks_hidden_size: Optional[int] = None,
        use_layer_norm: bool = True,
        affine_layer_norm: bool = True,
    ):
        super().__init__()

        self.encoder = TransformerEncoder(
            units_stream_size=units_stream_size,
            transformer_num_layers=transformer_num_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_key_size=transformer_key_size,
            transformer_value_size=transformer_value_size,
            resblocks_num_before=resblocks_num_before,
            resblocks_num_after=resblocks_num_after,
            resblocks_hidden_size=resblocks_hidden_size,
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )
        self.decoder = TransformerDecoder(
            units_stream_size=units_stream_size,
            transformer_num_layers=transformer_num_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_key_size=transformer_key_size,
            transformer_value_size=transformer_value_size,
            resblocks_num_before=resblocks_num_before,
            resblocks_num_after=resblocks_num_after,
            resblocks_hidden_size=resblocks_hidden_size,
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(enc_output, tgt, src_mask, tgt_mask)
        return dec_output


class AdaNorm(nn.Module):
    def __init__(self, adanorm_scale: float = 1.0, eps: float = 1e-5):
        self._adanorm_scale = adanorm_scale
        self._eps = eps

    def forward(self, input: torch.Tensor):
        mean = input.mean(-1, keepdim=True)
        std = input.std(-1, keepdim=True)
        input = input - mean
        mean = input.mean(-1, keepdim=True)
        graNorm = (1 / 10 * (input - mean) / (std + self._eps)).detach()
        input_norm = (input - input * graNorm) / (std + self._eps)
        return input_norm * self._adanorm_scale


class ToVector(nn.Module):
    def __init__(
        self,
        units_stream_size: int,
        units_hidden_sizes: Sequence[int],
        vector_stream_size: int,
        use_layer_norm: bool = True,
        affine_layer_norm: bool = False,
    ):
        super().__init__()

        self._units_stream_size = units_stream_size
        self._units_hidden_sizes = units_hidden_sizes
        self._vector_stream_size = vector_stream_size
        self._use_layer_norm = use_layer_norm

        hidden_size = units_hidden_sizes[0]

        mod_layers = [nn.ReLU()]
        if use_layer_norm:
            mod_layers.insert(
                0, nn.LayerNorm(units_stream_size, elementwise_affine=affine_layer_norm)
            )
        self._pre_layer = nn.Sequential(*mod_layers)
        self._hidden_layer = _layer_init(nn.Linear(units_stream_size, hidden_size))
        self._gate_layer = _layer_init(nn.Linear(units_stream_size, 1))

        self.out_ln = nn.LayerNorm(hidden_size, elementwise_affine=affine_layer_norm)
        self.out_lin = _layer_init(nn.Linear(hidden_size, vector_stream_size))

    def forward(self, entity_embeddings: torch.Tensor) -> torch.Tensor:
        x = entity_embeddings

        pre = self._pre_layer(x)
        gate = self._gate_layer(pre).softmax(-2)
        hidden = self._hidden_layer(pre)

        x = (gate.transpose(-2, -1) @ hidden).squeeze(-2)

        if self._use_layer_norm:
            x = self.out_ln(x)
        x = F.relu(x)
        return self.out_lin(x)


class GatingType(Enum):
    NONE = 0
    GLOBAL = 1
    POINTWISE = 2


class VectorMerge(nn.Module):
    def __init__(
        self,
        input_sizes: Dict[str, Optional[int]],
        output_size: int,
        gating_type: GatingType = GatingType.GLOBAL,
        use_layer_norm: bool = True,
        affine_layer_norm: bool = False,
    ):
        super().__init__()

        if not input_sizes:
            raise ValueError("input_sizes cannot be empty")

        self._input_sizes = input_sizes
        self._output_size = output_size
        self._gating_type = gating_type
        self._use_layer_norm = use_layer_norm

        self._linear_layers = nn.ModuleDict(
            {
                input_name: _layer_init(nn.Linear(input_size, output_size))
                for input_name, input_size in input_sizes.items()
            }
        )

        if gating_type == GatingType.GLOBAL:
            gate_size = 1
        elif gating_type == GatingType.POINTWISE:
            gate_size = output_size

        if gating_type != GatingType.NONE:
            gate_init = partial(
                _layer_init, std=0.005, bias_value=0, init_func=nn.init.normal_
            )
            self.gate_size = gate_size
            self._gate_layers = nn.ModuleDict(
                {
                    input_name: gate_init(
                        nn.Linear(input_size, len(input_sizes) * gate_size)
                    )
                    for input_name, input_size in input_sizes.items()
                }
            )

        if use_layer_norm:
            self._layernorms = nn.ModuleDict(
                {
                    input_name: nn.LayerNorm(
                        input_size, elementwise_affine=affine_layer_norm
                    )
                    for input_name, input_size in input_sizes.items()
                }
            )

    def _compute_gate(
        self,
        inputs_to_gate: Sequence[torch.Tensor],
        init_gate: Sequence[Tuple[str, torch.Tensor]],
    ) -> torch.Tensor:
        gate = torch.stack([self._gate_layers[name](y) for name, y in init_gate])
        gate = torch.sum(gate, dim=0)
        gate = gate.reshape(*gate.shape[:-1], len(inputs_to_gate), self.gate_size)
        gate = F.softmax(gate, dim=-2)
        return [gate[..., i, :] for i in range(gate.shape[-2])]

    def _encode(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[Sequence[Tuple[str, torch.Tensor]], Sequence[torch.Tensor]]:
        gates, outputs = [], []
        for name in self._input_sizes:
            feature = inputs[name]
            if self._use_layer_norm:
                feature = self._layernorms[name](feature)
            feature = F.relu(feature)
            gates.append((name, feature))
            outputs.append(self._linear_layers[name](feature))
        return gates, outputs

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        gates, outputs = self._encode(inputs)
        if len(outputs) == 1:
            # Special case of 1-D inputs that do not need any gating.
            output = outputs[0]
        elif self._gating_type == GatingType.NONE:
            output = torch.stack(outputs).sum(dim=0)
        else:
            gate = self._compute_gate(outputs, gates)
            data = [g * d for g, d in zip(gate, outputs)]
            output = sum(data)
        return output


class PointerLogits(nn.Module):
    def __init__(
        self,
        query_input_size: int,
        keys_input_size: int,
        num_layers_query: int = 1,
        num_layers_keys: int = 3,
        key_size: int = 64,
        use_layer_norm: bool = True,
        affine_layer_norm: bool = False,
    ):
        super().__init__()

        self.query_mlp = MLP(
            [query_input_size]
            + [query_input_size for _ in range(max(0, num_layers_query - 1))]
            + [key_size],
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )
        self.keys_mlp = MLP(
            [keys_input_size]
            + [keys_input_size for _ in range(max(0, num_layers_keys - 1))]
            + [key_size],
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )
        self.denom = 1 / math.sqrt(key_size)

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        query = self.query_mlp(query)
        keys = self.keys_mlp(keys)
        logits = keys @ query.transpose(-2, -1) * self.denom
        return logits


class GLU(nn.Module):
    def __init__(
        self,
        input_size: int,
        gate_size: int,
        output_size: int = None,
        use_layer_norm: bool = False,
    ):
        super().__init__()

        self.input_size = input_size
        self.gate_size = gate_size
        self.output_size = output_size or input_size

        self.gate_mlp = MLP(
            [self.gate_size, self.input_size], use_layer_norm=use_layer_norm
        )
        self.out_mlp = MLP(
            [self.input_size, self.output_size], use_layer_norm=use_layer_norm
        )

    def forward(self, input: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # The gate value is a learnt function of the input.
        gate = torch.sigmoid(self.gate_mlp(context))
        # Gate the input and return an output of desired size.
        return self.out_mlp(gate * input)
