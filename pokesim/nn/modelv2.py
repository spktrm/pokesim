import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from pokesim.data import (
    NUM_ABILITIES,
    NUM_BOOSTS,
    NUM_HP_BUCKETS,
    NUM_ITEMS,
    NUM_MOVES,
    NUM_PSEUDOWEATHER,
    NUM_SIDE_CONDITIONS,
    NUM_SPECIES,
    NUM_STATUS,
    NUM_TERRAIN,
    NUM_VOLATILE_STATUS,
    NUM_WEATHER,
)

from pokesim.structs import ModelOutput
from pokesim.utils import _legal_log_policy, _legal_policy

from pokesim.nn.helpers import _layer_init
from pokesim.nn.modules import (
    GatingType,
    PointerLogits,
    Resnet,
    ToVector,
    Transformer,
    VectorMerge,
    MLP,
)

# from pokemb.mod import PokEmb


PADDING_TOKEN = -1
UNKNOWN_TOKEN = -2
SWITCH_TOKEN = -3


class Model(nn.Module):
    def __init__(
        self,
        entity_size: int = 32,
        vector1_size: int = 8,
        vector2_size: int = 128,
        scale: int = 4,
        use_layer_norm: bool = False,
        affine_layer_norm: bool = False,
    ):
        super().__init__()

        entity_size *= scale
        vector1_size *= scale
        vector2_size *= scale

        self.entity_size = entity_size
        self.vector1_size = vector1_size
        self.vector2_size = vector2_size

        side_token = torch.zeros(18, dtype=torch.long).view(-1, 6)
        side_token[-1] = 1
        self.register_buffer("side_token", side_token)

        switch_tokens = SWITCH_TOKEN * torch.ones(6, dtype=torch.long)
        self.register_buffer("switch_tokens", switch_tokens)

        # self.pokemb = PokEmb(
        #     gen=3,
        #     use_layer_norm=use_layer_norm,
        #     output_size=entity_size,
        #     include_mlp=True,
        # )

        self.species_onehot = nn.Embedding.from_pretrained(torch.eye(NUM_SPECIES + 2))
        self.species_lin = _layer_init(
            nn.Linear(self.species_onehot.weight.shape[-1], entity_size)
        )

        self.item_onehot = nn.Embedding.from_pretrained(torch.eye(NUM_ITEMS + 2))
        self.item_lin = _layer_init(
            nn.Linear(self.item_onehot.weight.shape[-1], entity_size)
        )

        self.ability_onehot = nn.Embedding.from_pretrained(torch.eye(NUM_ABILITIES + 2))
        self.ability_lin = _layer_init(
            nn.Linear(self.ability_onehot.weight.shape[-1], entity_size)
        )

        self.moves_onehot = nn.Embedding.from_pretrained(torch.eye(NUM_MOVES + 3))
        self.moveset_lin = _layer_init(
            nn.Linear(self.moves_onehot.weight.shape[-1], entity_size)
        )
        self.moves_lin = _layer_init(
            nn.Linear(self.moves_onehot.weight.shape[-1], entity_size)
        )

        self.hp_onehot = nn.Embedding.from_pretrained(torch.eye(NUM_HP_BUCKETS + 1))
        self.hp_lin = _layer_init(
            nn.Linear(self.hp_onehot.weight.shape[-1], entity_size)
        )

        self.status_onehot = nn.Embedding.from_pretrained(torch.eye(NUM_STATUS + 1))
        self.status_lin = _layer_init(
            nn.Linear(self.status_onehot.weight.shape[-1], entity_size)
        )

        self.active_onehot = nn.Embedding.from_pretrained(torch.eye(2))
        self.active_lin = _layer_init(
            nn.Linear(self.active_onehot.weight.shape[-1], entity_size)
        )

        self.fainted_onehot = nn.Embedding.from_pretrained(torch.eye(2))
        self.fainted_lin = _layer_init(
            nn.Linear(self.fainted_onehot.weight.shape[-1], entity_size)
        )

        # self.entity_transformer = Transformer(
        #     units_stream_size=entity_size,
        #     transformer_num_layers=1,
        #     transformer_num_heads=2,
        #     transformer_key_size=entity_size // 2,
        #     transformer_value_size=entity_size // 2,
        #     resblocks_num_before=2,
        #     resblocks_num_after=2,
        #     resblocks_hidden_size=entity_size // 2,
        #     use_layer_norm=use_layer_norm,
        # )

        # self.side_onehot = _layer_init(nn.Embedding(2, entity_size))
        # self.public_onehot = _layer_init(nn.Embedding(2, entity_size))

        self.boosts_onehot = nn.Embedding.from_pretrained(torch.eye(13))
        boosts_size = NUM_BOOSTS + 12 * NUM_BOOSTS

        self.spikes_onehot = nn.Embedding.from_pretrained(torch.eye(4)[..., 1:])
        self.tspikes_onehot = nn.Embedding.from_pretrained(torch.eye(3)[..., 1:])
        self.volatile_status_onehot = nn.Embedding.from_pretrained(
            torch.eye(NUM_VOLATILE_STATUS + 1)[..., 1:]
        )

        side_condition_size = NUM_SIDE_CONDITIONS + 2 + 3
        self.context_linear = _layer_init(
            nn.Linear(
                boosts_size + NUM_VOLATILE_STATUS + side_condition_size, vector1_size
            )
        )
        self.context_mlp = MLP(
            [vector1_size, vector1_size], use_layer_norm=use_layer_norm
        )

        self.pseudoweathers_onehot = nn.Embedding.from_pretrained(
            torch.eye(NUM_PSEUDOWEATHER + 1)[..., 1:]
        )
        self.weathers_onehot = nn.Embedding.from_pretrained(
            torch.eye(NUM_WEATHER + 1)[..., 1:]
        )
        self.terrain_onehot = nn.Embedding.from_pretrained(
            torch.eye(NUM_TERRAIN + 1)[..., 1:]
        )
        field_size = NUM_PSEUDOWEATHER + NUM_WEATHER + NUM_TERRAIN
        self.field_linear = _layer_init(nn.Linear(field_size, vector1_size))

        self.entities_to_vector = ToVector(
            entity_size,
            [2 * entity_size],
            vector2_size,
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )

        self.side_merge = VectorMerge(
            {"entities": vector2_size, "context": vector1_size},
            vector2_size,
            gating_type=GatingType.NONE,
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )

        self.torso_merge = VectorMerge(
            {
                "my_side": vector2_size,
                "opp_side": vector2_size,
                "private_info": vector2_size,
                "field": vector1_size,
                # "action": vector_size,
                # "turn": vector_size // 4,
            },
            vector2_size,
            gating_type=GatingType.POINTWISE,
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )
        self.torso_resnet = Resnet(
            input_size=vector2_size,
            num_resblocks=2,
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )

        self.state_resnet = Resnet(
            input_size=vector2_size,
            num_resblocks=2,
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )
        self.user_resnet = Resnet(
            input_size=entity_size,
            num_resblocks=2,
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )
        self.action_resnet = Resnet(
            input_size=entity_size,
            num_resblocks=2,
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )
        self.action_merge = VectorMerge(
            {
                "state": vector2_size,
                "action": entity_size,
                "user": entity_size,
            },
            vector2_size,
            gating_type=GatingType.POINTWISE,
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )
        self.action_score = MLP(
            [vector2_size, vector2_size, 1],
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )
        # self.action_score = PointerLogits(
        #     vector2_size,
        #     2 * entity_size,
        #     num_layers_query=1,
        #     num_layers_keys=3,
        #     key_size=entity_size,
        #     use_layer_norm=use_layer_norm,
        # )

        self.value_resnet = Resnet(
            input_size=vector2_size,
            num_resblocks=2,
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )
        self.value_mlp = MLP(
            [vector2_size, vector2_size, 1],
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )

    def embed_teams(self, teams: torch.Tensor) -> torch.Tensor:
        teamsp1 = teams + 1
        teamsp2 = teams + 2
        teamsp3 = teams + 3

        species_token = teamsp2[..., 0]
        species_onehot = self.species_onehot(species_token)
        species_embedding = self.species_lin(species_onehot)

        item_token = teamsp2[..., 1]
        item_onehot = self.item_onehot(item_token)
        item_embedding = self.item_lin(item_onehot)

        ability_token = teamsp2[..., 2]
        ability_onehot = self.ability_onehot(ability_token)
        ability_embedding = self.ability_lin(ability_onehot)

        move_tokens = teamsp3[..., -4:]
        moveset_onehot = self.moves_onehot(move_tokens).sum(-2) / 4
        moveset_embedding = self.moveset_lin(moveset_onehot)

        # hp_value = teams[..., 3, None].float() / 1024
        hp_bucket = torch.sqrt(teams[..., 3]).to(torch.long)
        hp_onehot = self.hp_onehot(hp_bucket)
        hp_embedding = self.hp_lin(hp_onehot)

        active_token = teams[..., 4]
        active_onehot = self.active_onehot(active_token)
        active_embedding = self.active_lin(active_onehot)

        fainted_token = teams[..., 5]
        fainted_onehot = self.fainted_onehot(fainted_token)
        fainted_embedding = self.fainted_lin(fainted_onehot)

        status_token = teamsp1[..., 6]
        status_onehot = self.status_onehot(status_token)
        status_embedding = self.status_lin(status_onehot)

        embeddings = (
            species_embedding
            + item_embedding
            + ability_embedding
            + moveset_embedding
            + hp_embedding
            + fainted_embedding
            + status_embedding
        )

        return (embeddings, active_embedding)

    def encode_side_conditions(
        self, side_conditions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        other = side_conditions > 0
        spikes = self.spikes_onehot(side_conditions[..., 9])
        tspikes = self.tspikes_onehot(side_conditions[..., 13])
        encoding = torch.cat((other, spikes, tspikes), -1)
        return encoding

    def encode_volatile_status(
        self, volatile_status: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        volatile_status_id = volatile_status[..., 0]
        volatile_status_level = volatile_status[..., 1]
        encoding = self.volatile_status_onehot(volatile_status_id + 1).sum(-2)
        return encoding

    def encode_boosts(self, boosts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        boosts_onehot = self.boosts_onehot(boosts + 6)
        boosts_onehot = torch.cat((boosts_onehot[..., :6], boosts_onehot[..., 7:]), -1)
        boosts_scaled = torch.sign(boosts) * torch.sqrt(abs(boosts))
        encoding = torch.cat((boosts_onehot.flatten(-2), boosts_scaled), -1)
        return encoding

    def embed_field(self, field: torch.Tensor) -> torch.Tensor:
        field_id = field[..., 0]
        field_min_durr = field[..., 1]
        field_max_durr = field[..., 2]

        pseudoweathers = field_id[..., :3]
        pseudoweathers_onehot = self.pseudoweathers_onehot(pseudoweathers + 1).sum(-2)

        weather = field_id[..., 3]
        weather_onehot = self.weathers_onehot(weather + 1)

        terrain = field_id[..., 4]
        terrain_onehot = self.terrain_onehot(terrain + 1)

        encoding = torch.cat(
            (pseudoweathers_onehot, weather_onehot, terrain_onehot), -1
        )
        return self.field_linear(encoding)

    def forward(
        self,
        turn: torch.Tensor,
        active_moveset: torch.Tensor,
        teams: torch.Tensor,
        side_conditions: torch.Tensor,
        volatile_status: torch.Tensor,
        boosts: torch.Tensor,
        field: torch.Tensor,
        legal: torch.Tensor,
        history: torch.Tensor,
    ):
        entity_embeddings, active_embeddings = self.embed_teams(teams[:, :, -1])
        mask = teams[..., -1, :, :, 0] != PADDING_TOKEN

        active_token = teams[..., -1, :, :, 4]
        # entity_embeddings = self.entity_transformer(
        #     entity_embeddings + active_embeddings,
        #     torch.ones_like(entity_embeddings[..., 0]).bool(),
        # )

        active_weight = active_token.unsqueeze(-2).float()
        actives_embedding = (active_weight @ entity_embeddings).squeeze(-2)
        entities_embedding = self.entities_to_vector(
            entity_embeddings + active_embeddings, mask
        )

        side_conditions_encoding = self.encode_side_conditions(side_conditions)
        volatile_status_encoding = self.encode_volatile_status(volatile_status)
        boosts_encoding = self.encode_boosts(boosts)

        context_encoding = torch.cat(
            (side_conditions_encoding, volatile_status_encoding, boosts_encoding),
            dim=-1,
        )
        context_embedding = self.context_mlp(self.context_linear(context_encoding))

        side_embedding = self.side_merge(
            {
                "entities": entities_embedding[..., :2, :],
                "context": context_embedding[:, :, -1],
            }
        )

        field_embedding = self.embed_field(field)

        private_info = torch.abs(
            entities_embedding[:, :, 0] - entities_embedding[:, :, 1]
        )

        state_embedding = self.torso_merge(
            {
                # actives
                "my_side": side_embedding[..., 0, :],
                "opp_side": side_embedding[..., 1, :],
                "private_info": private_info,
                # field
                "field": field_embedding[..., -1, :],
            },
        )
        state_embedding = self.torso_resnet(state_embedding)

        active_movesets = (
            (active_weight @ teams[:, :, -1].float()).long()[..., -4:].squeeze(-2)
        )
        active_movesets = (
            torch.cat(
                (
                    active_movesets,
                    self.switch_tokens.view(1, 1, 1, -1).expand(
                        *active_movesets.shape[:2], 3, 6
                    ),
                ),
                dim=-1,
            )
            + 3
        )

        action_tokens = active_movesets[..., 0, :]
        action_embeddings = self.moves_lin(self.moves_onehot(action_tokens))
        entity_embeddings = entity_embeddings[..., 0, :, :]
        user_embeddings = torch.cat(
            (actives_embedding[..., :1, :].expand(-1, -1, 4, -1), entity_embeddings),
            dim=-2,
        )

        state_query = (
            self.state_resnet(state_embedding).unsqueeze(-2).expand(-1, -1, 10, -1)
        )
        action_keys = self.action_resnet(action_embeddings)
        user_keys = self.user_resnet(user_embeddings)
        action_embeddings = self.action_merge(
            {
                "state": state_query,
                "action": action_embeddings,
                "user": user_embeddings,
            }
        )

        logits = self.action_score(action_embeddings).flatten(2)
        # logits = self.action_score(action_query, action_embeddings).flatten(2)

        policy = _legal_policy(logits, legal)
        log_policy = _legal_log_policy(logits, legal)

        value_hidden = self.value_resnet(state_embedding)
        value = self.value_mlp(value_hidden)

        return ModelOutput(
            logits=logits, policy=policy, log_policy=log_policy, value=value
        )
