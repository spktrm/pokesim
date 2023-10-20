import torch
import math


def discretize_probs(probs, disc):
    n = 1 / disc

    # Sort the probabilities in descending order and also get the indices
    sorted_probs, indices = torch.sort(probs, descending=True, dim=-1)

    # Calculate the rounded up values, making sure they're multiples of 1/n
    rounded_probs = torch.ceil(sorted_probs * disc) / disc

    # Calculate the amount needed to reach 1.0
    correction = 1.0 - rounded_probs.sum(dim=-1)

    for i in reversed(range(probs.shape[-1])):
        new_probs = (rounded_probs[..., i] + correction).clamp(min=0)
        correction = (correction + rounded_probs[..., i]).clamp(max=0)
        rounded_probs[..., i] = new_probs

    # Restore the original order of elements
    final_probs = torch.zeros_like(probs).scatter_(-1, indices, rounded_probs)

    return final_probs


# Test the function with a tensor of action probabilities and n_disc
probs = torch.randn(8, 5)

mask = torch.randint(0, 2, probs.shape)
mask[..., 0] = 1

probs = torch.where(mask == 1, probs, float("-inf")).softmax(-1)

n_disc = 16  # you can adjust n_disc based on your requirement

print(probs)
discretized_probs = discretize_probs(probs, n_disc)

print(discretized_probs)

print(discretized_probs.sum(-1))
