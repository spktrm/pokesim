import torch


def _legal_policy(logits: torch.Tensor, legal_actions: torch.Tensor) -> torch.Tensor:
    return torch.where(legal_actions, logits, float("-inf")).softmax(-1)


def _legal_log_policy(
    logits: torch.Tensor, legal_actions: torch.Tensor
) -> torch.Tensor:
    return torch.where(
        legal_actions,
        torch.where(legal_actions, logits, float("-inf")).log_softmax(-1),
        0,
    )
