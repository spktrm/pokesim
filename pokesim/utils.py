import torch
import torch.nn as nn

from tabulate import tabulate


class SGDTowardsModel:
    def __init__(self, params_target: nn.Module, params: nn.Module, lr: float):
        self._params_target = params_target
        self._params = params
        self._lr = lr

    def step(self):
        target_state_dict = self._params_target.state_dict()
        state_dict = self._params.state_dict()

        for (key, param_target), (_, param) in zip(
            target_state_dict.items(), state_dict.items()
        ):
            target_state_dict[key] = self._lr * param + (1 - self._lr) * param_target

        self._params_target.load_state_dict(target_state_dict)


def _print_params(model: nn.Module):
    trainable_params_count = sum(
        x.numel() for x in model.parameters() if x.requires_grad
    )
    params_count = sum(x.numel() for x in model.parameters())
    print(f"""Total Trainable Params: {trainable_params_count:,}""")
    print(f"""Total Params: {params_count:,}""")

    rows = []
    for name, mod in model.named_modules():
        if (
            name
            and name.count(".") == 0
            or (isinstance(mod, nn.Embedding) and mod.weight.requires_grad)
        ):
            mod_params_count = sum(
                x.numel() for x in mod.parameters() if x.requires_grad
            )
            if mod_params_count == 0:
                continue
            rows.append(
                [
                    name,
                    f"{mod_params_count:,}",
                    f"{100 * mod_params_count / trainable_params_count:.2f}",
                ]
            )
    print(tabulate(rows, headers=["name", "param_count", "param_ratio"]))

    return trainable_params_count, params_count


@torch.jit.script_if_tracing
def _legal_policy(logits: torch.Tensor, legal_actions: torch.Tensor) -> torch.Tensor:
    """A soft-max policy that respects legal_actions."""
    # Fiddle a bit to make sure we don't generate NaNs or Inf in the middle.
    l_min = logits.min(dim=-1, keepdim=True).values
    logits = torch.where(legal_actions, logits, l_min)
    logits -= logits.max(dim=-1, keepdim=True).values
    logits *= legal_actions
    exp_logits = torch.where(
        legal_actions, torch.exp(logits), 0
    )  # Illegal actions become 0.
    exp_logits_sum = torch.sum(exp_logits, dim=-1, keepdim=True)
    return exp_logits / exp_logits_sum


@torch.jit.script_if_tracing
def _legal_log_policy(
    logits: torch.Tensor, legal_actions: torch.Tensor
) -> torch.Tensor:
    """Return the log of the policy on legal action, 0 on illegal action."""
    # logits_masked has illegal actions set to -inf.
    logits_masked = logits + torch.log(legal_actions)
    max_legal_logit = logits_masked.max(dim=-1, keepdim=True).values
    logits_masked = logits_masked - max_legal_logit
    # exp_logits_masked is 0 for illegal actions.
    exp_logits_masked = torch.exp(logits_masked)

    baseline = torch.log(torch.sum(exp_logits_masked, dim=-1, keepdim=True))
    # Subtract baseline from logits. We do not simply return
    #     logits_masked - baseline
    # because that has -inf for illegal actions, or
    #     legal_actions * (logits_masked - baseline)
    # because that leads to 0 * -inf == nan for illegal actions.
    log_policy = torch.multiply(legal_actions, (logits - max_legal_logit - baseline))
    return log_policy


def _threshold(
    policy: torch.Tensor, mask: torch.Tensor, threshold: float = 0.03
) -> torch.Tensor:
    """Remove from the support the actions 'a' where policy(a) < threshold."""
    if threshold <= 0:
        return policy

    mask = mask * (
        # Values over the threshold.
        (policy >= threshold)
        +
        # Degenerate case is when policy is less than threshold *everywhere*.
        # In that case we just keep the policy as-is.
        (torch.max(policy, dim=-1, keepdim=True).values < threshold)
    )
    return mask * policy / torch.sum(mask * policy, dim=-1, keepdim=True)


def _discretize(policy: torch.Tensor, n_disc: float = 16) -> torch.Tensor:
    """Round all action probabilities to a multiple of 1/self.discretize."""

    # Sort the probabilities in descending order and also get the indices
    sorted_probs, indices = torch.sort(policy, descending=True, dim=-1)

    # Calculate the rounded up values, making sure they're multiples of 1/n
    rounded_probs = torch.ceil(sorted_probs * n_disc) / n_disc

    # Calculate the amount needed to reach 1.0
    correction = 1.0 - rounded_probs.sum(dim=-1)

    for i in reversed(range(policy.shape[-1])):
        new_probs = (rounded_probs[..., i] + correction).clamp(min=0)
        correction = (correction + rounded_probs[..., i]).clamp(max=0)
        rounded_probs[..., i] = new_probs

    # Restore the original order of elements
    final_probs = torch.zeros_like(policy).scatter_(-1, indices, rounded_probs)

    return final_probs


@torch.jit.script_if_tracing
def finetune(policy: torch.Tensor, mask: torch.Tensor):
    policy = _threshold(policy, mask)
    # policy = _discretize(policy)
    return policy
