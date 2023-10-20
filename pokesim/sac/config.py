from typing import List

import numpy as np


class AdamConfig:
    """Adam optimizer related params."""

    b1: float = 0.0
    b2: float = 0.999
    eps: float = 10e-8


class SACConfig:
    """Configuration parameters for the RNaDSolver."""

    actor_device = "cpu"
    learner_device = "cuda"

    # The batch size to use when learning/improving parameters.
    batch_size: int = 8
    # The learning rate for `params`.
    learning_rate: float = 5e-5
    # The config related to the ADAM optimizer used for updating `params`.
    adam: AdamConfig = AdamConfig()
    # All gradients values are clipped to [-clip_gradient, clip_gradient].
    clip_gradient: float = 10000

    gamma = 0.99
    gae_param = 0.95

    clip_param = 0.1
    # Weight of value function loss in the total loss.
    vf_coeff = 0.5
    # Weight of entropy bonus in the total loss.
    entropy_coeff = 0.01
    tau = 1e-2

    forward_batch_size: int = 512
    backward_batch_size: int = 256
