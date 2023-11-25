import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence

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
from pokesim.nn.modules import GLU, GatingType, PointerLogits, ResNet, VectorMerge, MLP


PADDING_TOKEN = -1
UNKNOWN_TOKEN = -2
SWITCH_TOKEN = -3


kl_loss = lambda x, y: F.kl_div(x, y, log_target=True, reduction="none").sum(-1)

cosine_sim_loss = lambda x, y: -(F.normalize(x, dim=-1) * F.normalize(y, dim=-1)).mean(
    -1
)


def gumbel_softmax_zero_temp(
    tensor: torch.Tensor, buckets: int = 16, hard: bool = False
):
    flat = tensor.reshape(-1, buckets)
    sm = flat.softmax(-1).view_as(tensor)
    if hard:
        argmax = torch.argmax(flat, dim=-1)
        onehot = F.one_hot(argmax, buckets)
        shaped = onehot.view_as(tensor)
        return shaped + sm.detach() - sm
    else:
        return sm


class Model(nn.Module):
    def __init__(
        self,
        entity_size: int = 128,
        vector_size: int = 384,
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
        self.species_mlp = MLP(
            [entity_size, entity_size], use_layer_norm=use_layer_norm
        )

        self.item_onehot = _layer_init(nn.Embedding(NUM_ITEMS + 2, entity_size))
        self.item_glu = GLU(entity_size, entity_size, use_layer_norm=use_layer_norm)

        self.ability_onehot = _layer_init(nn.Embedding(NUM_ABILITIES + 2, entity_size))
        self.ability_glu = GLU(entity_size, entity_size, use_layer_norm=use_layer_norm)

        self.moves_onehot = _layer_init(nn.Embedding(NUM_MOVES + 3, entity_size))
        self.moves_glu = GLU(entity_size, entity_size, use_layer_norm=use_layer_norm)
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

        self.units_enc = MLP([entity_size, entity_size], use_layer_norm=use_layer_norm)

        self.action_glu1 = GLU(entity_size, entity_size, use_layer_norm=use_layer_norm)
        # self.action_mu = MLP([entity_size, entity_size], use_layer_norm=use_layer_norm)
        # self.action_log_var = MLP(
        #     [entity_size, entity_size], use_layer_norm=use_layer_norm
        # )
        # self.action_glu2 = GLU(entity_size, entity_size, use_layer_norm=use_layer_norm)

        self.turn_embedding = _layer_init(nn.Embedding(64, vector_size // 4))

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
                boosts_size + NUM_VOLATILE_STATUS + side_condition_size,
                vector_size // 4,
            )
        )
        self.context_mlp = MLP(
            [
                vector_size // 4,
                vector_size // 4,
            ],
            use_layer_norm=use_layer_norm,
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
        self.field_linear = _layer_init(nn.Linear(field_size, vector_size // 4))

        # self.dynamics_func = nn.GRU(entity_size, vector_size)

        self.torso_merge = VectorMerge(
            {
                "my_private_active": entity_size,
                "my_public_active": entity_size,
                "opp_public_active": entity_size,
                "my_private_reserve": entity_size,
                "my_public_reserve": entity_size,
                "opp_public_reserve": entity_size,
                "my_context": vector_size // 4,
                "opp_context": vector_size // 4,
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

        self.my_action_query_resnet = ResNet(vector_size, use_layer_norm=use_layer_norm)
        self.my_action_pointer = PointerLogits(
            vector_size,
            entity_size,
            key_size=entity_size // 2,
            num_layers_keys=1,
            num_layers_query=3,
            use_layer_norm=use_layer_norm,
        )

        self.opp_action_query_resnet = ResNet(
            vector_size, use_layer_norm=use_layer_norm
        )
        self.opp_action_pointer = PointerLogits(
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

    def forward_species(self, token: torch.Tensor) -> torch.Tensor:
        # return self.pokemb.forward_species(x)
        embedding = self.species_onehot(token)
        return self.species_mlp(embedding)

    def forward_moves(self, token: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # return self.pokemb.forward_moves(x)
        embedding = F.relu(self.moves_onehot(token))
        return self.moves_glu(embedding, context)

    def forward_items(self, token: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        embedding = F.relu(self.item_onehot(token))
        return self.item_glu(embedding, context)

    def forward_abilities(
        self, token: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        embedding = F.relu(self.ability_onehot(token))
        return self.ability_glu(embedding, context)

    def pred_curr_state(
        self,
        prev_state: torch.Tensor,
        prev_entity_embeddings: torch.Tensor,
        prev_action_history: torch.Tensor,
    ) -> torch.Tensor:
        T, B, *_ = prev_entity_embeddings.shape

        action_hist_mask = prev_action_history[..., 0] >= 0
        action_hist_len = action_hist_mask.sum(-1).flatten(0, 1).clamp(min=1)
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

        action_mu = self.action_mu(action_embeddings)
        action_log_var = self.action_log_var(action_embeddings)

        action_embeddings = action_mu + torch.randn_like(action_log_var) * torch.exp(
            0.5 * action_log_var
        )
        action_embeddings = self.action_glu2(
            action_embeddings, self.side_onehot(action_side)
        )
        og_shape = action_embeddings.shape
        action_embeddings = action_embeddings.view(-1, 16).softmax(-1).view(*og_shape)
        action_embeddings = action_embeddings.flatten(0, 1).transpose(0, 1)

        packed_input = pack_padded_sequence(
            action_embeddings,
            action_hist_len.cpu(),
            enforce_sorted=False,
        )
        _, ht = self.dynamics_func(packed_input, prev_state.flatten(0, 1).unsqueeze(0))
        next_state = ht[-1].view(T, B, -1)

        return next_state

    def embed_state_history(
        self, step_embeddings: torch.Tensor, hidden_state: torch.Tensor
    ) -> torch.Tensor:
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

        embeddings = self.units_enc(
            species_onehot
            + item_onehot
            + moveset_onehot
            + ability_onehot
            + self.entity_onehots(onehots)
        )
        return embeddings

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
        return self.field_linear(encoding)

    def kl_balance(
        self, posterior: torch.Tensor, prior: torch.Tensor, alpha: float = 0.8
    ):
        T, B, *_ = posterior.shape

        kl1 = kl_loss(prior, posterior.detach())
        kl2 = kl_loss(posterior, prior.detach())

        kl1 = kl1.view(T, B, -1).mean(-1)
        kl2 = kl2.view(T, B, -1).mean(-1)

        forward_dynamics_loss = alpha * kl1 + (1 - alpha) * kl2

        return forward_dynamics_loss

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

        private_entities = gumbel_softmax_zero_temp(
            entity_embeddings[..., -1, :1, :, :], hard=True
        )
        public_entities = gumbel_softmax_zero_temp(
            entity_embeddings[..., -1, 1:, :, :], hard=False
        )
        entity_embeddings = torch.cat((private_entities, public_entities), dim=2)

        active_token = teams[:, :, -1, ..., 4]
        active_mask = (active_token + (teams[:, :, -1, ..., 0] == PADDING_TOKEN)).bool()

        active_weight = (
            torch.where(active_mask, 0, -1e9).softmax(-1).unsqueeze(-2).float()
        )

        reserve_weight = (
            torch.where(active_mask, -1e9, 0).softmax(-1).unsqueeze(-2).float()
        )

        active_embeddings = (active_weight @ entity_embeddings).squeeze(-2)

        private_active_exists = torch.any(active_mask[..., 0, :], dim=-1).unsqueeze(-1)
        private_active_embedding = private_active_exists * active_embeddings[..., 0, :]

        public_active_exists = torch.any(active_mask[..., 1:, :], dim=-1).unsqueeze(-1)
        public_active_embeddings = public_active_exists * active_embeddings[..., 1:, :]

        reserve_embeddings = (reserve_weight @ entity_embeddings).squeeze(-2)
        private_reserve_embedding = reserve_embeddings[..., 0, :]
        public_reserve_embeddings = reserve_embeddings[..., 1:, :]

        side_conditions_encoding = self.encode_side_conditions(side_conditions)
        volatile_status_encoding = self.encode_volatile_status(volatile_status)
        boosts_encoding = self.encode_boosts(boosts)

        context_encoding = torch.cat(
            (side_conditions_encoding, volatile_status_encoding, boosts_encoding),
            dim=-1,
        )
        context_encoding = self.context_mlp(self.context_linear(context_encoding))
        context_encoding = gumbel_softmax_zero_temp(
            context_encoding[..., -1, :], hard=True
        )

        field_embedding = self.embed_field(field)
        field_embedding = gumbel_softmax_zero_temp(
            field_embedding[..., -1, :], hard=True
        )

        # turn_embedding = self.turn_embedding(
        #     turn[:, :, -1].clamp(min=0, max=self.turn_embedding.weight.shape[0] - 1)
        # )

        state_embeddings = self.torso_merge(
            {
                "my_private_active": private_active_embedding,
                "my_public_active": public_active_embeddings[..., 0, :],
                "opp_public_active": public_active_embeddings[..., 1, :],
                "my_private_reserve": private_reserve_embedding,
                "my_public_reserve": public_reserve_embeddings[..., 0, :],
                "opp_public_reserve": public_reserve_embeddings[..., 1, :],
                "my_context": context_encoding[..., 0, :],
                "opp_context": context_encoding[..., 1, :],
                "field": field_embedding,
                # "action": vector_size,
                # "turn": vector_size // 4,
            },
        )

        # T, B, H, *_ = state_embeddings.shape
        # state_embeddings = (
        #     state_embeddings.view(T, B, H, -1, 16).log_softmax(-1).view(T, B, H, -1)
        # )

        # prev_state = state_embeddings[:, :, 0]
        # prev_action = history[:, :, -1]
        # prev_entities = entity_embeddings[:, :, 0]
        # curr_state = state_embeddings[:, :, 1]
        curr_state = state_embeddings

        # pred_state = self.pred_curr_state(
        #     torch.exp(prev_state), prev_entities, prev_action
        # )
        # pred_state = pred_state.view(T, B, -1, 16).log_softmax(-1).view(T, B, -1)

        # curr_state, hidden_state = self.embed_state_history(curr_state, hidden_state)

        # forward_dynamics_loss = self.kl_balance(
        #     pred_state.view(T, B, -1, 16), curr_state.view(T, B, -1, 16).log_softmax(-1)
        # )

        curr_state, hidden_state = self.embed_state_history(curr_state, hidden_state)

        # curr_state = torch.exp(curr_state)

        my_active_moveset = torch.cat(
            (
                active_moveset[:, :, -1],
                self.switch_tokens[(None,) * 2].expand(*active_moveset.shape[:2], -1),
            ),
            dim=-1,
        )

        opp_active_moveset = (
            (active_weight @ teams[:, :, -1].float()).squeeze(-2).long()[:, :, -1, -4:]
        )
        opp_active_moveset = torch.cat(
            (
                opp_active_moveset,
                self.switch_tokens[(None,) * 2].expand(*active_moveset.shape[:2], -1),
            ),
            dim=-1,
        )

        my_action_target_embeddings = torch.cat(
            (
                public_active_embeddings[..., 1:, :].expand(-1, -1, 4, -1),
                entity_embeddings[:, :, 0],
            ),
            dim=-2,
        )
        opp_action_target_embeddings = torch.cat(
            (
                private_active_embedding.unsqueeze(-2).expand(-1, -1, 4, -1),
                entity_embeddings[:, :, -1],
            ),
            dim=-2,
        )

        my_action_embeddings = self.forward_moves(
            my_active_moveset + 3,
            private_active_embedding.unsqueeze(-2).expand(-1, -1, 10, -1),
        )
        opp_action_embeddings = self.forward_moves(
            opp_active_moveset + 3,
            public_active_embeddings[..., 1:, :].expand(-1, -1, 10, -1),
        )

        my_action_embeddings = self.action_glu1(
            my_action_embeddings, my_action_target_embeddings
        )
        opp_action_embeddings = self.action_glu1(
            opp_action_embeddings, opp_action_target_embeddings
        )

        my_action_query = self.my_action_query_resnet(curr_state).unsqueeze(-2)
        my_logits = self.my_action_pointer(
            my_action_query, my_action_embeddings
        ).flatten(2)

        opp_action_query = self.opp_action_query_resnet(curr_state).unsqueeze(-2)
        opp_logits = self.opp_action_pointer(
            opp_action_query, opp_action_embeddings
        ).flatten(2)

        logits = my_logits.unsqueeze(-1) @ opp_logits.unsqueeze(-2)
        logits = logits.mean(-1)

        policy = _legal_policy(logits, legal)
        log_policy = _legal_log_policy(logits, legal)

        value_hidden = self.value_resnet(curr_state)
        value = self.value_mlp(value_hidden)

        return ModelOutput(
            logits=logits,
            policy=policy,
            log_policy=log_policy,
            value=value,
            hidden_state=hidden_state,
            # forward_dynamics_loss=forward_dynamics_loss,
        )
