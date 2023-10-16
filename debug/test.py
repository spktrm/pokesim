import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.autograd.profiler as profiler


from pokesim.model import Model

model = Model()


def get_example(T: int, B: int, device: str):
    return (
        torch.zeros(T, B, 8, dtype=torch.long, device=device),
        torch.zeros(T, B, 4, dtype=torch.long, device=device),
        torch.zeros(T, B, 8, 3, 6, 11, dtype=torch.long, device=device),
        torch.zeros(T, B, 8, 2, 15, dtype=torch.long, device=device),
        torch.zeros(T, B, 8, 2, 20, 2, dtype=torch.long, device=device),
        torch.zeros(T, B, 8, 2, 7, dtype=torch.long, device=device),
        torch.zeros(T, B, 8, 5, 3, dtype=torch.long, device=device),
        torch.ones(T, B, 12, dtype=torch.bool, device=device),
        torch.ones(T, B, 8, 4, 3, dtype=torch.long, device=device),
        torch.ones(T, B, 8, dtype=torch.bool, device=device),
    )


device = "cpu"
model = model.to(device)
jit_model = torch.jit.trace(model, get_example(32, 32, device))
