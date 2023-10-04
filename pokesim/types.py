import torch
import numpy as np

from typing import Dict, Any, Literal

TensorType = np.ndarray | torch.Tensor
TensorDict = Dict[str, TensorType | Any]

OpponentPolicy = Literal["maxdmg", "default", "random"]
