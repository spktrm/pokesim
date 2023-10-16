import torch


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


@torch.jit.script_if_tracing
def _threshold(
    policy: torch.Tensor, mask: torch.Tensor, threshold: float = 0.05
) -> torch.Tensor:
    """Remove from the support the actions 'a' where policy(a) < threshold."""
    mask = mask * (
        # Values over the threshold.
        (policy >= threshold)
        +
        # Degenerate case is when policy is less than threshold *everywhere*.
        # In that case we just keep the policy as-is.
        (torch.max(policy, dim=-1, keepdim=True).values < threshold)
    )
    return mask * policy / torch.sum(mask * policy, dim=-1, keepdim=True)
