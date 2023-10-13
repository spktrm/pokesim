import math
import torch
import torch.nn as nn

from typing import Dict, Callable
from pokesim.config import RNaDConfig

from pokesim.structs import ModelOutput


def optimized_forward(
    module: nn.Module,
    inputs: Dict[str, torch.Tensor],
    config: RNaDConfig,
    batch_size: int = 1024,
) -> ModelOutput:
    results = []

    first_key = next(iter(inputs.keys()))
    T, B, *_ = inputs[first_key].shape

    inputs = {k: v.view(T * B, 1, *v.shape[2:]) for k, v in inputs.items()}

    for i in range(math.ceil(T * B / batch_size)):
        minibatch = {
            k: v[i * batch_size : (i + 1) * batch_size].to(
                config.learner_device, non_blocking=True
            )
            for k, v in inputs.items()
        }
        results.append(module(**minibatch))

    return ModelOutput(*map(lambda x: torch.cat(x).view(T, B, -1), zip(*results)))
