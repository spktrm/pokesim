import math
import torch
import torch.nn as nn
import numpy as np

from typing import Dict

from pokesim.sac.config import SACConfig
from pokesim.structs import ModelOutput


def optimized_forward(
    module: nn.Module,
    inputs: Dict[str, torch.Tensor],
    config: SACConfig,
) -> ModelOutput:
    results = []

    first_key = next(iter(inputs.keys()))
    T, B, *_ = inputs[first_key].shape

    inputs = {k: v.view(T * B, 1, *v.shape[2:]) for k, v in inputs.items()}

    batch_size = config.forward_batch_size
    for i in range(math.ceil(T * B / batch_size)):
        minibatch = {
            k: v[i * batch_size : (i + 1) * batch_size].to(
                config.learner_device, non_blocking=True
            )
            for k, v in inputs.items()
        }
        results.append([t.detach().cpu() for t in module(**minibatch)])

    return ModelOutput(*map(lambda x: torch.cat(x).view(T, B, -1), zip(*results)))


def gae_advantages(
    rewards: np.ndarray,
    terminal_masks: np.ndarray,
    values: np.ndarray,
    discount: float,
    gae_param: float,
):
    """Use Generalized Advantage Estimation (GAE) to compute advantages.

    As defined by eqs. (11-12) in PPO paper arXiv: 1707.06347. Implementation uses
    key observation that A_{t} = delta_t + gamma*lambda*A_{t+1}.

    Args:
      rewards: array shaped (actor_steps, num_agents), rewards from the game
      terminal_masks: array shaped (actor_steps, num_agents), zeros for terminal
                      and ones for non-terminal states
      values: array shaped (actor_steps, num_agents), values estimated by critic
      discount: RL discount usually denoted with gamma
      gae_param: GAE parameter usually denoted with lambda

    Returns:
      advantages: calculated advantages shaped (actor_steps, num_agents)
    """
    assert rewards.shape[0] + 1 == values.shape[0], (
        "One more value needed; Eq. "
        "(12) in PPO paper requires "
        "V(s_{t+1}) for delta_t"
    )
    advantages = []
    gae = np.zeros_like(values[0])
    for t in reversed(range(len(rewards))):
        # Masks used to set next state value to 0 for terminal states.
        value_diff = discount * values[t + 1] * terminal_masks[t] - values[t]
        delta = rewards[t] + value_diff
        # Masks[t] used to ensure that values before and after a terminal state
        # are independent of each other.
        gae = delta + discount * gae_param * terminal_masks[t] * gae
        advantages.append(gae)
    advantages = advantages[::-1]
    return np.stack(advantages)
