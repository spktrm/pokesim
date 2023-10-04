import math

import pandas as pd
import numpy as np

import torch
import torch.nn as nn


def multi_hot_encode(indices_tensor, num_classes):
    # Get the shape of the tensor, and create a new shape for the multi-hot encoded tensor
    shape = list(indices_tensor.shape[:-1]) + [num_classes]
    # Create a tensor filled with zeros of the new shape
    multi_hot = torch.zeros(
        shape, device=indices_tensor.device, dtype=indices_tensor.dtype
    )
    # Use scatter_add to add the ones to the multi_hot tensor at the specified indices
    multi_hot.scatter_add_(
        -1, indices_tensor, torch.ones_like(indices_tensor, dtype=indices_tensor.dtype)
    )
    return multi_hot


def faster_bmm(a: torch.Tensor, b: torch.Tensor):
    expand_shape = (math.prod(a.shape[:-2]),) + b.shape
    lht = b.expand(expand_shape)
    return torch.bmm(a.view(-1, *a.shape[-2:]), lht).view(*a.shape[:-1], -1)


class EntityEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        moves_df = pd.read_csv(
            "./dist/src/gen9randombattle/moves.csv", index_col="species"
        )
        abilities_df = pd.read_csv(
            "./dist/src/gen9randombattle/abilities.csv", index_col="species"
        )
        items_df = pd.read_csv(
            "./dist/src/gen9randombattle/items.csv", index_col="species"
        )

        abilities_init = abilities_df.values.astype(np.float32)
        items_init = items_df.values.astype(np.float32)
        moves_init = moves_df.values.astype(np.float32)

        embeddings = torch.eye(moves_df.shape[0] + 1)[:, 1:]
        self.species_onehot = nn.Embedding.from_pretrained(embeddings)
        self.all_abilities = nn.Embedding.from_pretrained(
            torch.from_numpy(abilities_init)
        )
        self.abilities_onehot = nn.Embedding.from_pretrained(
            torch.eye(abilities_init.shape[1] + 1)[..., 1:]
        )
        self.all_items = nn.Embedding.from_pretrained(torch.from_numpy(items_init))
        self.items_onehot = nn.Embedding.from_pretrained(
            torch.eye(items_init.shape[1] + 1)[..., 1:]
        )

        self.all_moves = nn.Embedding.from_pretrained(torch.from_numpy(moves_init))
        self.moves_onehot = lambda x: multi_hot_encode(x, moves_init.shape[1] + 1)[
            ..., 1:
        ]
        self.moves_shape = moves_init.shape

    def forward_species(self, token: torch.Tensor, mask: torch.Tensor):
        # Assume shape is [..., 6]
        mask = mask.unsqueeze(-1)
        onehot = self.species_onehot(token)
        unknown = torch.ones_like(onehot) - onehot.sum(-2, keepdim=True)
        unknown /= unknown.sum(-1, keepdim=True).clamp(min=1)
        return torch.where(mask, onehot, unknown)

    def forward_ability(
        self,
        species_embedding: torch.Tensor,
        ability_token: torch.Tensor,
        mask: torch.Tensor,
    ):
        mask = mask.unsqueeze(-1)
        known = self.abilities_onehot((ability_token - 1).clamp(min=0))
        unknown = faster_bmm(species_embedding, self.all_abilities.weight)
        unknown = unknown / unknown.sum(-1, keepdim=True).clamp(min=1)
        return torch.where(mask, known, unknown)

    def forward_item(
        self,
        species_embedding: torch.Tensor,
        item_token: torch.Tensor,
        mask: torch.Tensor,
    ):
        mask = mask.unsqueeze(-1)
        known = self.items_onehot((item_token - 1).clamp(min=0))
        unknown = faster_bmm(species_embedding, self.all_items.weight)
        unknown /= unknown.sum(-1, keepdim=True).clamp(min=1)
        return torch.where(mask, known, unknown)

    def forward_moveset(
        self,
        species_embedding: torch.Tensor,
        move_tokens: torch.Tensor,
        mask: torch.Tensor,
    ):
        known = self.moves_onehot(move_tokens)
        all_unknown = faster_bmm(species_embedding, self.all_moves.weight)
        unknown = all_unknown - known
        unknown /= unknown.sum(-1, keepdim=True).clamp(min=1)
        num_missing = 4 - (known > 0).sum(-1, keepdim=True)
        return torch.where(mask, known + num_missing * unknown, 4 * unknown)

    def forward(
        self,
        species_token: torch.Tensor,
        ability_token: torch.Tensor,
        item_token: torch.Tensor,
        move_tokens: torch.Tensor,
    ):
        species_mask = species_token > 0
        species_embedding = self.forward_species(species_token, species_mask)

        ability_mask = ability_token > 0
        ability_embedding = self.forward_ability(
            species_embedding, ability_token, ability_mask
        )

        item_mask = item_token > 0
        item_embedding = self.forward_item(species_embedding, item_token, item_mask)

        move_mask = (species_mask & (move_tokens.sum(-1) != 0)).unsqueeze(-1)
        move_embedding = self.forward_moveset(species_embedding, move_tokens, move_mask)

        return species_embedding, ability_embedding, item_embedding, move_embedding
