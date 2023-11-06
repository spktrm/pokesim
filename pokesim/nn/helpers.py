import math

import torch.nn as nn


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
