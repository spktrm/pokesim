import math
import torch
import torch.nn as nn
import numpy as np

import jax
import chex

from jax import lax
from jax import numpy as jnp
from jax import tree_util as tree

from functools import partial
from typing import Any, Dict, List, Tuple

from pokesim.rnad.config import RNaDConfig
from pokesim.structs import ModelOutput


def optimized_forward(
    module: nn.Module,
    inputs: Dict[str, torch.Tensor],
    config: RNaDConfig,
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


class EntropySchedule:
    """An increasing list of steps where the regularisation network is updated.

    Example
      EntropySchedule([3, 5, 10], [2, 4, 1])
      =>   [0, 3, 6, 11, 16, 21, 26, 10]
            | 3 x2 |      5 x4     | 10 x1
    """

    def __init__(self, *, sizes: List[int], repeats: List[int]):
        """Constructs a schedule of entropy iterations.

        Args:
          sizes: the list of iteration sizes.
          repeats: the list, parallel to sizes, with the number of times for each
            size from `sizes` to repeat.
        """
        try:
            if len(repeats) != len(sizes):
                raise ValueError("`repeats` must be parallel to `sizes`.")
            if not sizes:
                raise ValueError("`sizes` and `repeats` must not be empty.")
            if any([(repeat <= 0) for repeat in repeats]):
                raise ValueError("All repeat values must be strictly positive")
            if repeats[-1] != 1:
                raise ValueError(
                    "The last value in `repeats` must be equal to 1, "
                    "ince the last iteration size is repeated forever."
                )
        except ValueError as e:
            raise ValueError(
                f"Entropy iteration schedule: repeats ({repeats}) and sizes"
                f" ({sizes})."
            ) from e

        schedule = [0]
        for size, repeat in zip(sizes, repeats):
            schedule.extend([schedule[-1] + (i + 1) * size for i in range(repeat)])

        self.schedule = np.array(schedule, dtype=np.int32)

    def __call__(self, learner_step: int) -> Tuple[float, bool]:
        """Entropy scheduling parameters for a given `learner_step`.

        Args:
          learner_step: The current learning step.

        Returns:
          alpha: The mixing weight (from [0, 1]) of the previous policy with
            the one before for computing the intrinsic reward.
          update_target_net: A boolean indicator for updating the target network
            with the current network.
        """

        # The complexity below is because at some point we might go past
        # the explicit schedule, and then we'd need to just use the last step
        # in the schedule and apply the logic of
        # ((learner_step - last_step) % last_iteration) == 0)

        # The schedule might look like this:
        # X----X-------X--X--X--X--------X
        # learner_step | might be here ^    |
        # or there     ^                    |
        # or even past the schedule         ^

        # We need to deal with two cases below.
        # Instead of going for the complicated conditional, let's just
        # compute both and then do the A * s + B * (1 - s) with s being a bool
        # selector between A and B.

        # 1. assume learner_step is past the schedule,
        #    ie schedule[-1] <= learner_step.
        last_size = self.schedule[-1] - self.schedule[-2]
        last_start = (
            self.schedule[-1]
            + (learner_step - self.schedule[-1]) // last_size * last_size
        )
        # 2. assume learner_step is within the schedule.
        start = jnp.amax(self.schedule * (self.schedule <= learner_step))
        finish = jnp.amin(
            self.schedule * (learner_step < self.schedule),
            initial=self.schedule[-1],
            where=(learner_step < self.schedule),
        )
        size = finish - start

        # Now select between the two.
        beyond = self.schedule[-1] <= learner_step  # Are we past the schedule?
        iteration_start = last_start * beyond + start * (1 - beyond)
        iteration_size = last_size * beyond + size * (1 - beyond)

        update_target_net = jnp.logical_and(
            learner_step > 0,
            jnp.sum(learner_step == iteration_start + iteration_size - 1),
        )
        alpha = jnp.minimum(
            (2.0 * (learner_step - iteration_start)) / iteration_size, 1.0
        )

        return (
            float(alpha.item()),
            bool(update_target_net.item()),
        )  # pytype: disable=bad-return-type  # jax-types


def _player_others(
    player_ids: np.ndarray, valid: np.ndarray, player: int
) -> np.ndarray:
    """A vector of 1 for the current player and -1 for others.

    Args:
      player_ids: Tensor [...] containing player ids (0 <= player_id < N).
      valid: Tensor [...] containing whether these states are valid.
      player: The player id as int.

    Returns:
      player_other: is 1 for the current player and -1 for others [..., 1].
    """
    chex.assert_equal_shape((player_ids, valid))
    current_player_tensor = (player_ids == player).astype(
        jnp.int32
    )  # pytype: disable=attribute-error  # numpy-scalars

    res = 2 * current_player_tensor - 1
    res = res * valid
    return jnp.expand_dims(res, axis=-1)


def _policy_ratio(
    pi: np.ndarray, mu: np.ndarray, actions_oh: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """Returns a ratio of policy pi/mu when selecting action a.

    By convention, this ratio is 1 on non valid states
    Args:
      pi: the policy of shape [..., A].
      mu: the sampling policy of shape [..., A].
      actions_oh: a one-hot encoding of the current actions of shape [..., A].
      valid: 0 if the state is not valid and else 1 of shape [...].

    Returns:
      pi/mu on valid states and 1 otherwise. The shape is the same
      as pi, mu or actions_oh but without the last dimension A.
    """
    chex.assert_equal_shape((pi, mu, actions_oh))
    chex.assert_shape((valid,), actions_oh.shape[:-1])

    def _select_action_prob(pi):
        return jnp.sum(actions_oh * pi, axis=-1, keepdims=False) * valid + (1 - valid)

    pi_actions_prob = _select_action_prob(pi)
    mu_actions_prob = _select_action_prob(mu)
    return pi_actions_prob / mu_actions_prob


def _where(pred, true_data, false_data):
    """Similar to jax.where but treats `pred` as a broadcastable prefix."""

    def _where_one(t, f):
        chex.assert_equal_rank((t, f))
        # Expand the dimensions of pred if true_data and false_data are higher rank.
        p = jnp.reshape(pred, pred.shape + (1,) * (len(t.shape) - len(pred.shape)))
        return jnp.where(p, t, f)

    return tree.tree_map(_where_one, true_data, false_data)


def _has_played(valid: np.ndarray, player_id: np.ndarray, player: int) -> np.ndarray:
    """Compute a mask of states which have a next state in the sequence."""
    chex.assert_equal_shape((valid, player_id))

    def _loop_has_played(carry, x):
        valid, player_id = x
        chex.assert_equal_shape((valid, player_id))

        our_res = jnp.ones_like(player_id)
        opp_res = carry
        reset_res = jnp.zeros_like(carry)

        our_carry = carry
        opp_carry = carry
        reset_carry = jnp.zeros_like(player_id)

        # pyformat: disable
        return _where(
            valid,
            _where((player_id == player), (our_carry, our_res), (opp_carry, opp_res)),
            (reset_carry, reset_res),
        )
        # pyformat: enable

    _, result = lax.scan(
        f=_loop_has_played,
        init=jnp.zeros_like(player_id[-1]),
        xs=(valid, player_id),
        reverse=True,
    )
    return result


# V-Trace
#
# Custom implementation of VTrace to handle trajectories having a mix of
# different player steps. The standard rlax.vtrace can't be applied here
# out of the box because a trajectory could look like '121211221122'.


@partial(jax.jit, static_argnames=["eta", "lambda_", "c", "rho", "gamma"])
def v_trace(
    v: np.ndarray,
    valid: np.ndarray,
    player_id: np.ndarray,
    acting_policy: np.ndarray,
    merged_policy: np.ndarray,
    merged_log_policy: np.ndarray,
    player_others: np.ndarray,
    actions_oh: np.ndarray,
    reward: np.ndarray,
    player: int,
    # Scalars below.
    eta: float,
    lambda_: float,
    c: float,
    rho: float,
    gamma: float = 1.0,
) -> Tuple[Any, Any, Any]:
    """Custom VTrace for trajectories with a mix of different player steps."""

    has_played = _has_played(valid, player_id, player)

    policy_ratio = _policy_ratio(merged_policy, acting_policy, actions_oh, valid)
    inv_mu = _policy_ratio(
        jnp.ones_like(merged_policy), acting_policy, actions_oh, valid
    )

    eta_reg_entropy = (
        -eta
        * jnp.sum(merged_policy * merged_log_policy, axis=-1)
        * jnp.squeeze(player_others, axis=-1)
    )
    eta_log_policy = -eta * merged_log_policy * player_others

    @chex.dataclass(frozen=True)
    class LoopVTraceCarry:
        """The carry of the v-trace scan loop."""

        reward: np.ndarray
        # The cumulated reward until the end of the episode. Uncorrected (v-trace).
        # Gamma discounted and includes eta_reg_entropy.
        reward_uncorrected: np.ndarray
        next_value: np.ndarray
        next_v_target: np.ndarray
        importance_sampling: np.ndarray

    init_state_v_trace = LoopVTraceCarry(
        reward=jnp.zeros_like(reward[-1]),
        reward_uncorrected=jnp.zeros_like(reward[-1]),
        next_value=jnp.zeros_like(v[-1]),
        next_v_target=jnp.zeros_like(v[-1]),
        importance_sampling=jnp.ones_like(policy_ratio[-1]),
    )

    def _loop_v_trace(carry: LoopVTraceCarry, x) -> Tuple[LoopVTraceCarry, Any]:
        (
            cs,
            player_id,
            v,
            reward,
            eta_reg_entropy,
            valid,
            inv_mu,
            actions_oh,
            eta_log_policy,
        ) = x

        reward_uncorrected = reward + gamma * carry.reward_uncorrected + eta_reg_entropy
        discounted_reward = reward + gamma * carry.reward

        # V-target:
        our_v_target = (
            v
            + jnp.expand_dims(jnp.minimum(rho, cs * carry.importance_sampling), axis=-1)
            * (
                jnp.expand_dims(reward_uncorrected, axis=-1)
                + gamma * carry.next_value
                - v
            )
            + lambda_
            * jnp.expand_dims(jnp.minimum(c, cs * carry.importance_sampling), axis=-1)
            * gamma
            * (carry.next_v_target - carry.next_value)
        )

        opp_v_target = jnp.zeros_like(our_v_target)
        reset_v_target = jnp.zeros_like(our_v_target)

        # Learning output:
        our_learning_output = (
            v
            + eta_log_policy  # value
            + actions_oh  # regularisation
            * jnp.expand_dims(inv_mu, axis=-1)
            * (
                jnp.expand_dims(discounted_reward, axis=-1)
                + gamma
                * jnp.expand_dims(carry.importance_sampling, axis=-1)
                * carry.next_v_target
                - v
            )
        )

        opp_learning_output = jnp.zeros_like(our_learning_output)
        reset_learning_output = jnp.zeros_like(our_learning_output)

        # State carry:
        our_carry = LoopVTraceCarry(
            reward=jnp.zeros_like(carry.reward),
            next_value=v,
            next_v_target=our_v_target,
            reward_uncorrected=jnp.zeros_like(carry.reward_uncorrected),
            importance_sampling=jnp.ones_like(carry.importance_sampling),
        )
        opp_carry = LoopVTraceCarry(
            reward=eta_reg_entropy + cs * discounted_reward,
            reward_uncorrected=reward_uncorrected,
            next_value=gamma * carry.next_value,
            next_v_target=gamma * carry.next_v_target,
            importance_sampling=cs * carry.importance_sampling,
        )
        reset_carry = init_state_v_trace

        # Invalid turn: init_state_v_trace and (zero target, learning_output)
        # pyformat: disable
        return _where(
            valid,  # pytype: disable=bad-return-type  # numpy-scalars
            _where(
                (player_id == player),
                (our_carry, (our_v_target, our_learning_output)),
                (opp_carry, (opp_v_target, opp_learning_output)),
            ),
            (reset_carry, (reset_v_target, reset_learning_output)),
        )
        # pyformat: enable

    _, (v_target, learning_output) = lax.scan(
        f=_loop_v_trace,
        init=init_state_v_trace,
        xs=(
            policy_ratio,
            player_id,
            v,
            reward,
            eta_reg_entropy,
            valid,
            inv_mu,
            actions_oh,
            eta_log_policy,
        ),
        reverse=True,
    )

    return v_target, has_played, learning_output
