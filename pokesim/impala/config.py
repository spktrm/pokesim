import numpy as np

from typing import List
from dataclasses import dataclass


@dataclass
class AdamConfig:
    """Adam optimizer related params."""

    b1: float = 0.0
    b2: float = 0.999
    eps: float = 10e-8


@dataclass
class NerdConfig:
    """Nerd related params."""

    beta: float = 2.0
    clip: float = 10_000


@dataclass
class ImpalaConfig:
    """Configuration parameters for the RNaDSolver."""

    actor_device = "cpu"
    learner_device = "cuda"

    # The batch size to use when learning/improving parameters.
    batch_size: int = 4
    # The learning rate for `params`.
    learning_rate: float = 5e-5
    # The config related to the ADAM optimizer used for updating `params`.
    adam: AdamConfig = AdamConfig()
    # All gradients values are clipped to [-clip_gradient, clip_gradient].
    clip_gradient: float = 10000

    nerd: NerdConfig = NerdConfig()
    c_vtrace: float = 1.0
    rho: float = np.inf

    tau: float = 1e-2
