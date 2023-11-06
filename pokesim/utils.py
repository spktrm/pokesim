import torch
import torch.nn as nn

from tabulate import tabulate


class SGDTowardsModel:
    def __init__(self, params_target: nn.Module, params: nn.Module, lr: float):
        self._params_target = params_target
        self._params = params
        self._lr = lr

    @torch.no_grad()
    def step(self):
        for param_target, param in zip(
            self._params_target.parameters(), self._params.parameters()
        ):
            if param.requires_grad:
                new_val = self._lr * param + (1 - self._lr) * param_target
                param_target.copy_(new_val)


def _print_params(model: nn.Module):
    from pokesim.nn.model import MLP, PointerLogits, ResNet, ToVector, VectorMerge

    trainable_params_count = sum(
        x.numel() for x in model.parameters() if x.requires_grad
    )
    params_count = sum(x.numel() for x in model.parameters())
    print(f"""Total Trainable Params: {trainable_params_count:,}""")
    print(f"""Total Params: {params_count:,}""")

    rows = []
    for name, mod in model.named_modules():
        if (
            isinstance(
                mod,
                (MLP, nn.LSTM, nn.GRU, ToVector, VectorMerge, ResNet, PointerLogits),
            )
            and name.count(".") == 0
        ):
            mod_params_count = sum(
                x.numel() for x in mod.parameters() if x.requires_grad
            )
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
    return torch.where(legal_actions, logits, float("-inf")).softmax(-1)


@torch.jit.script_if_tracing
def _legal_log_policy(
    logits: torch.Tensor, legal_actions: torch.Tensor
) -> torch.Tensor:
    return torch.where(
        legal_actions,
        torch.where(legal_actions, logits, float("-inf")).log_softmax(-1),
        0,
    )


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
    policy = _discretize(policy)
    return policy
