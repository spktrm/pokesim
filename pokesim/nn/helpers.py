import math
from typing import Callable
import torch

import torch.nn as nn


def _layer_init(
    layer: nn.Module,
    mean: float = None,
    std: float = None,
    bias_value: float = None,
    init_func: Callable = None,
):
    if hasattr(layer, "weight"):
        if isinstance(layer, nn.Embedding):
            init_func = init_func or nn.init.normal_
        elif isinstance(layer, nn.Linear):
            init_func = init_func or nn.init.trunc_normal_
        if std is None:
            n = getattr(layer, "num_embeddings", None) or getattr(layer, "in_features")
            std = 1 / math.sqrt(n)
        init_func(layer.weight, mean=(mean or 0), std=std)
    if hasattr(layer, "data"):
        init_func = init_func or nn.init.normal_
        if std is None:
            n = getattr(layer, "data").shape[-1]
            std = 1 / math.sqrt(n)
        init_func(layer.data, mean=(mean or 0), std=std)
    if hasattr(layer, "bias") and getattr(layer, "bias", None) is not None:
        nn.init.constant_(layer.bias, val=(bias_value or 0))
    return layer


class RMSNorm(nn.Module):
    def __init__(self, d, p=-1.0, eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
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


def get_layer_norm(input_shape: int):
    return RMSNorm(input_shape)
