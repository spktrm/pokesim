import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence

from typing import Tuple

from pokemb import PokEmb

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
    GLU,
    VAE,
    GatingType,
    PointerLogits,
    ResNet,
    VectorMerge,
    MLP,
)


PADDING_TOKEN = -1
UNKNOWN_TOKEN = -2
SWITCH_TOKEN = -3


class Model(nn.Module):
    def __init__(
        self,
        entity_size: int = 128,
        vector_size: int = 512,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        self.entity_size = entity_size
        self.vector_size = vector_size

        side_token = torch.zeros(18, dtype=torch.long).view(-1, 6)
        side_token[-1] = 1
        self.register_buffer("side_token", side_token)

        switch_tokens = SWITCH_TOKEN * torch.ones(6, dtype=torch.long)
        self.register_buffer("switch_tokens", switch_tokens)

        # self.pokemb = PokEmb(9, output_size=entity_size)

        self.species_onehot = _layer_init(nn.Embedding(NUM_SPECIES + 2, entity_size))

        self.item_onehot = _layer_init(nn.Embedding(NUM_ITEMS + 2, entity_size))
        self.item_glu = GLU(entity_size, entity_size, use_layer_norm=use_layer_norm)
        # self.item_vae = VAE(entity_size, use_layer_norm=use_layer_norm)

        self.ability_onehot = _layer_init(nn.Embedding(NUM_ABILITIES + 2, entity_size))
        self.ability_glu = GLU(entity_size, entity_size, use_layer_norm=use_layer_norm)
        # self.ability_vae = VAE(entity_size, use_layer_norm=use_layer_norm)

        self.moves_onehot = _layer_init(nn.Embedding(NUM_MOVES + 3, entity_size))
        self.moves_glu = GLU(entity_size, entity_size, use_layer_norm=use_layer_norm)
        # self.moves_vae = VAE(entity_size, use_layer_norm=use_layer_norm)
        self.moveset_mlp = MLP(
            [entity_size, entity_size], use_layer_norm=use_layer_norm
        )

        self.hp_onehot = nn.Embedding.from_pretrained(torch.eye(NUM_HP_BUCKETS + 1))
        self.status_onehot = nn.Embedding.from_pretrained(torch.eye(NUM_STATUS + 1))
        self.active_onehot = nn.Embedding.from_pretrained(torch.eye(2))
        self.fainted_onehot = nn.Embedding.from_pretrained(torch.eye(2))
        self.entity_onehots = _layer_init(
            nn.Linear(
                self.hp_onehot.weight.shape[-1]
                + self.status_onehot.weight.shape[-1]
                + self.active_onehot.weight.shape[-1]
                + self.fainted_onehot.weight.shape[-1],
                entity_size,
            )
        )

        self.side_onehot = _layer_init(nn.Embedding(2, entity_size))
        # self.public_onehot = _layer_init(nn.Embedding(2, entity_size))

        self.units_enc = MLP(
            [entity_size, entity_size, entity_size], use_layer_norm=use_layer_norm
        )

        self.action_glu1 = GLU(entity_size, entity_size, use_layer_norm=use_layer_norm)
        self.action_glu2 = GLU(entity_size, entity_size, use_layer_norm=use_layer_norm)

        self.turn_embedding = _layer_init(nn.Embedding(64, vector_size // 4))

        self.boosts_onehot = nn.Embedding.from_pretrained(torch.eye(13))
        boosts_size = NUM_BOOSTS + 12 * NUM_BOOSTS
        self.boosts_linear = _layer_init(nn.Linear(boosts_size, vector_size))
        self.boosts_mlp = MLP(
            [vector_size, vector_size // 4], use_layer_norm=use_layer_norm
        )

        self.spikes_onehot = nn.Embedding.from_pretrained(torch.eye(4)[..., 1:])
        self.tspikes_onehot = nn.Embedding.from_pretrained(torch.eye(3)[..., 1:])
        self.volatile_status_onehot = nn.Embedding.from_pretrained(
            torch.eye(NUM_VOLATILE_STATUS + 1)[..., 1:]
        )
        self.volatile_status_linear = _layer_init(
            nn.Linear(NUM_VOLATILE_STATUS, vector_size)
        )
        self.volatile_status_mlp = MLP(
            [vector_size, vector_size // 4], use_layer_norm=use_layer_norm
        )

        side_condition_size = NUM_SIDE_CONDITIONS + 2 + 3
        self.side_condition_linear = _layer_init(
            nn.Linear(side_condition_size, vector_size)
        )
        self.side_condition_mlp = MLP(
            [vector_size, vector_size // 4], use_layer_norm=use_layer_norm
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
        self.field_linear = _layer_init(nn.Linear(field_size, vector_size))
        self.field_mlp = MLP(
            [vector_size, vector_size // 4], use_layer_norm=use_layer_norm
        )

        self.action_rnn = nn.GRU(
            input_size=entity_size,
            hidden_size=vector_size,
            num_layers=1,
        )

        self.side_merge = VectorMerge(
            {
                "active": entity_size,
                "reserve": entity_size,
                "side_conditions": vector_size // 4,
                "volatile_status": vector_size // 4,
                "boosts": vector_size // 4,
            },
            vector_size,
            gating_type=GatingType.POINTWISE,
            use_layer_norm=use_layer_norm,
        )

        self.torso_merge = VectorMerge(
            {
                "my_private_side": vector_size,
                "my_public_side": vector_size,
                "opp_public_side": vector_size,
                "field": vector_size // 4,
                # "action": vector_size,
                # "turn": vector_size // 4,
            },
            vector_size,
            gating_type=GatingType.POINTWISE,
            use_layer_norm=use_layer_norm,
        )

        self.state_rnn = nn.GRU(
            input_size=vector_size,
            hidden_size=vector_size,
            num_layers=1,
        )

        self.action_query_resnet = ResNet(vector_size, use_layer_norm=use_layer_norm)
        self.action_pointer = PointerLogits(
            vector_size,
            entity_size,
            key_size=entity_size // 2,
            num_layers_keys=1,
            num_layers_query=3,
            use_layer_norm=use_layer_norm,
        )

        self.value_resnet = ResNet(
            vector_size, use_layer_norm=use_layer_norm, num_resblocks=4
        )
        self.value_mlp = MLP([vector_size, 1], use_layer_norm=use_layer_norm)

    def forward_species(self, token: torch.Tensor):
        # return self.pokemb.forward_species(x)
        return self.species_onehot(token)

    def forward_moves(self, token: torch.Tensor, context: torch.Tensor):
        # return self.pokemb.forward_moves(x)
        embedding = F.relu(self.moves_onehot(token))
        gated = self.moves_glu(embedding, context)
        return gated  # self.moves_vae(gated)

    def forward_items(self, token: torch.Tensor, context: torch.Tensor):
        # return self.pokemb.forward_items(x)
        embedding = F.relu(self.item_onehot(token))
        gated = self.item_glu(embedding, context)
        return gated  # self.item_vae(gated)

    def forward_abilities(self, token: torch.Tensor, context: torch.Tensor):
        # return self.pokemb.forward_abilities(x)
        embedding = F.relu(self.ability_onehot(token))
        gated = self.ability_glu(embedding, context)
        return gated  # self.ability_vae(gated)

    def pred_curr_state(
        self,
        prev_state: torch.Tensor,
        prev_entity_embeddings: torch.Tensor,
        prev_action_history: torch.Tensor,
    ):
        T, B, *_ = prev_entity_embeddings.shape

        action_hist_mask = prev_action_history[..., 0] >= 0
        action_hist_len = action_hist_mask.sum(-1).flatten().clamp(min=1)
        action_side = prev_action_history[..., 0].clamp(min=0)
        action_user = prev_action_history[..., 1].clamp(min=0)
        action_target = prev_action_history[..., 2].clamp(min=0)
        action_move = prev_action_history[..., 3] + 3

        entity_embeddings_flat = prev_entity_embeddings.flatten(-3, -2)

        action_user_onehot = F.one_hot(action_user + 6, 18)
        action_user_embeddings = action_user_onehot.float() @ entity_embeddings_flat

        action_target_onehot = F.one_hot(action_target + 6, 18)
        action_target_embeddings = action_target_onehot.float() @ entity_embeddings_flat

        action_embeddings = self.forward_moves(action_move, action_user_embeddings)
        action_embeddings = self.action_glu1(
            action_embeddings, action_target_embeddings
        )
        action_embeddings = self.action_glu2(
            action_embeddings, self.side_onehot(action_side)
        )

        action_embeddings = action_embeddings.flatten(0, 1).transpose(0, 1)
        packed_input = pack_padded_sequence(
            F.gumbel_softmax(action_embeddings, tau=1, hard=True, dim=-1),
            action_hist_len.cpu(),
            enforce_sorted=False,
        )
        _, ht = self.action_rnn(packed_input, prev_state.flatten(0, 1).unsqueeze(0))
        action_history_embedding = ht[-1].view(T, B, -1)
        return action_history_embedding

    def embed_state_history(
        self, step_embeddings: torch.Tensor, hidden_state: torch.Tensor
    ):
        return self.state_rnn(step_embeddings, hidden_state)

    def embed_teams(self, teams: torch.Tensor) -> torch.Tensor:
        teamsp1 = teams + 1
        teamsp2 = teams + 2
        teamsp3 = teams + 3

        species_token = teamsp2[..., 0]
        species_onehot = self.forward_species(species_token)

        item_token = teamsp2[..., 1]
        item_onehot = self.forward_items(item_token, species_onehot)

        ability_token = teamsp2[..., 2]
        ability_onehot = self.forward_abilities(ability_token, species_onehot)

        move_tokens = teamsp3[..., -4:]
        moves_onehot = self.forward_moves(move_tokens, species_onehot.unsqueeze(-2))
        moves_weight = (
            torch.where((move_tokens - 3 - PADDING_TOKEN) == 0, -1e9, 0)
            .softmax(-1)
            .unsqueeze(-2)
        )
        moveset_onehot = self.moveset_mlp(moves_weight @ moves_onehot).squeeze(-2)

        # hp_value = teams[..., 3, None].float() / 1024
        hp_bucket = torch.sqrt(teams[..., 3]).to(torch.long)
        hp_onehot = self.hp_onehot(hp_bucket)

        active_token = teams[..., 4]
        active_onehot = self.active_onehot(active_token)

        fainted_token = teams[..., 5]
        fainted_onehot = self.fainted_onehot(fainted_token)

        status_token = teamsp1[..., 6]
        status_onehot = self.status_onehot(status_token)

        onehots = torch.cat(
            (hp_onehot, active_onehot, fainted_onehot, status_onehot), dim=-1
        )

        encodings = (
            species_onehot
            + item_onehot
            + ability_onehot
            + moveset_onehot
            + self.entity_onehots(onehots)
        )

        encodings = self.units_enc(encodings)
        return encodings

    def embed_side_conditions(
        self, side_conditions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        other = side_conditions > 0
        spikes = self.spikes_onehot(side_conditions[..., 9])
        tspikes = self.tspikes_onehot(side_conditions[..., 13])
        encoding = torch.cat((other, spikes, tspikes), -1)
        encoding = self.side_condition_linear(encoding)
        return self.side_condition_mlp(encoding)

    def embed_volatile_status(
        self, volatile_status: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        volatile_status_id = volatile_status[..., 0]
        volatile_status_level = volatile_status[..., 1]
        encoding = self.volatile_status_onehot(volatile_status_id + 1).sum(-2)
        encoding = self.volatile_status_linear(encoding)
        return self.volatile_status_mlp(encoding)

    def embed_boosts(self, boosts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        boosts_onehot = self.boosts_onehot(boosts + 6)
        boosts_onehot = torch.cat((boosts_onehot[..., :6], boosts_onehot[..., 7:]), -1)
        boosts_scaled = torch.sign(boosts) * torch.sqrt(abs(boosts))
        encoding = torch.cat((boosts_onehot.flatten(-2), boosts_scaled), -1)
        encoding = self.boosts_linear(encoding)
        return self.boosts_mlp(encoding)

    def get_hidden_state(self, batch_size: int = 1):
        return torch.zeros((self.state_rnn.num_layers, batch_size, self.vector_size))

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
        encoding = self.field_linear(encoding)
        return self.field_mlp(encoding)

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
        hidden_state: torch.Tensor = None,
    ):
        entity_embeddings = self.embed_teams(teams)
        active_token = teams[..., 4].unsqueeze(-2).float()
        active_count = active_token.sum(-1).clamp(min=1)

        reserve_weight = (
            torch.where(
                (teams[..., 4] + (teams[..., 0] == PADDING_TOKEN)).bool(), -1e9, 0
            )
            .softmax(-1)
            .unsqueeze(-2)
            .float()
        )

        active_embeddings = (active_token @ entity_embeddings).squeeze(-2)
        active_embeddings = active_embeddings / active_count

        reserve_embeddings = (reserve_weight @ entity_embeddings).squeeze(-2)

        side_conditions_embedding = self.embed_side_conditions(side_conditions)
        volatile_status_embedding = self.embed_volatile_status(volatile_status)
        boosts_embedding = self.embed_boosts(boosts)
        field_embedding = self.embed_field(field)

        # action_hist_embedding = self.embed_action_history(
        #     entity_embeddings[:, :, 0], history[:, :, -1]
        # )
        # turn_embedding = self.turn_embedding(
        #     turn[:, :, -1].clamp(min=0, max=self.turn_embedding.weight.shape[0] - 1)
        # )

        sides = self.side_merge(
            {
                "active": active_embeddings,
                "reserve": reserve_embeddings,
                "side_conditions": torch.cat(
                    (
                        side_conditions_embedding[..., :1, :].expand(-1, -1, -1, 2, -1),
                        side_conditions_embedding[..., 1:, :],
                    ),
                    dim=-2,
                ),
                "volatile_status": torch.cat(
                    (
                        volatile_status_embedding[..., :1, :].expand(-1, -1, -1, 2, -1),
                        volatile_status_embedding[..., 1:, :],
                    ),
                    dim=-2,
                ),
                "boosts": torch.cat(
                    (
                        boosts_embedding[..., :1, :].expand(-1, -1, -1, 2, -1),
                        boosts_embedding[..., 1:, :],
                    ),
                    dim=-2,
                ),
            }
        )

        (
            my_private_side_embedding,
            my_public_side_embedding,
            opp_public_side_embedding,
        ) = torch.chunk(sides, 3, -2)

        state_embeddings = self.torso_merge(
            {
                "my_private_side": my_private_side_embedding.squeeze(-2),
                "my_public_side": my_public_side_embedding.squeeze(-2),
                "opp_public_side": opp_public_side_embedding.squeeze(-2),
                "field": field_embedding,
                # "action": action_hist_embedding,
                # "turn": turn_embedding,
            }
        )

        prev_state_embedding = state_embeddings[:, :, 0]
        state_embedding = state_embeddings[:, :, 1]

        pred_state_embedding = self.pred_curr_state(
            prev_state_embedding, entity_embeddings[:, :, 0], history[:, :, -1]
        )

        forward_dynamics_loss = F.kl_div(
            F.log_softmax(pred_state_embedding, dim=-1),
            F.softmax(state_embedding.detach(), dim=-1),
            reduction="none",
        ).sum(-1)

        state_embedding, hidden_state = self.embed_state_history(
            state_embedding, hidden_state
        )

        active_moveset = torch.cat(
            (
                active_moveset[:, :, -1],
                self.switch_tokens[(None,) * 2].expand(*active_moveset.shape[:2], -1),
            ),
            dim=-1,
        )

        action_target_embeddings = torch.cat(
            (
                active_embeddings[..., -1, 2:, :].expand(-1, -1, 4, -1),
                entity_embeddings[:, :, -1, 0],
            ),
            dim=-2,
        )
        action_embeddings = self.forward_moves(
            active_moveset + 3,
            entity_embeddings[:, :, -1, 0, :1].expand(-1, -1, 10, -1),
        )
        action_embeddings = self.action_glu1(
            action_embeddings, action_target_embeddings
        )

        action_query = self.action_query_resnet(state_embedding).unsqueeze(-2)
        logits = self.action_pointer(action_query, action_embeddings).flatten(2)

        policy = _legal_policy(logits, legal)
        log_policy = _legal_log_policy(logits, legal)

        value_hidden = self.value_resnet(state_embedding)
        value = self.value_mlp(value_hidden)

        return ModelOutput(
            logits=logits,
            policy=policy,
            log_policy=log_policy,
            value=value,
            hidden_state=hidden_state,
            forward_dynamics_loss=forward_dynamics_loss,
        )
