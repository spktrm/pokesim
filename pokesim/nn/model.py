import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence

from typing import Tuple

from pokesim.data import (
    NUM_ABILITIES,
    NUM_BOOSTS,
    NUM_HISTORY,
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
    VectorMerge,
    VectorQuantizer,
    MLP,
    _USE_LAYER_NORM,
)


class Model(nn.Module):
    def __init__(self, entity_size: int = 128, vector_size: int = 256):
        super().__init__()

        self.entity_size = entity_size
        self.vector_size = vector_size

        side_token = torch.zeros(18, dtype=torch.long).view(-1, 6)
        side_token[-1] = 1
        self.register_buffer("side_token", side_token)

        self.species_onehot = _layer_init(nn.Embedding(NUM_SPECIES + 1, entity_size))
        self.item_onehot = _layer_init(nn.Embedding(NUM_ITEMS + 1, entity_size))
        self.ability_onehot = _layer_init(nn.Embedding(NUM_ABILITIES + 1, entity_size))
        self.moves_onehot = _layer_init(nn.Embedding(NUM_MOVES + 2, entity_size))
        self.hp_onehot = _layer_init(nn.Embedding(NUM_HP_BUCKETS + 1, entity_size))
        self.status_onehot = _layer_init(nn.Embedding(NUM_STATUS + 1, entity_size))
        self.active_onehot = _layer_init(nn.Embedding(2, entity_size))
        self.fainted_onehot = _layer_init(nn.Embedding(2, entity_size))
        self.side_onehot = _layer_init(nn.Embedding(2, entity_size))
        self.public_onehot = _layer_init(nn.Embedding(2, entity_size))

        self.unit_quantizer = VectorQuantizer(128, entity_size // 4)

        self.units_mlp = MLP([entity_size, entity_size], use_layer_norm=_USE_LAYER_NORM)
        self.moves_mlp = MLP([entity_size, entity_size], use_layer_norm=_USE_LAYER_NORM)

        self.species_dec = MLP([entity_size, NUM_SPECIES + 1])
        self.item_dec = MLP([entity_size, NUM_ITEMS + 1])
        self.ability_dec = MLP([entity_size, NUM_ABILITIES + 1])
        self.moveset_dec = MLP([entity_size, NUM_MOVES + 2])
        self.hp_dec = MLP([entity_size, NUM_HP_BUCKETS + 1])
        self.status_dec = MLP([entity_size, NUM_STATUS + 1])
        self.active_dec = MLP([entity_size, 2])
        self.fainted_dec = MLP([entity_size, 2])
        self.side_dec = MLP([entity_size, 2])
        self.public_dec = MLP([entity_size, 2])

        self.spikes_onehot = nn.Embedding.from_pretrained(torch.eye(4)[..., 1:])
        self.tspikes_onehot = nn.Embedding.from_pretrained(torch.eye(3)[..., 1:])
        self.volatile_status_onehot = nn.Embedding.from_pretrained(
            torch.eye(NUM_VOLATILE_STATUS + 1)[..., 1:]
        )
        self.boosts_onehot = nn.Embedding.from_pretrained(torch.eye(13))
        self.pseudoweathers_onehot = nn.Embedding.from_pretrained(
            torch.eye(NUM_PSEUDOWEATHER + 1)[..., 1:]
        )
        self.weathers_onehot = nn.Embedding.from_pretrained(
            torch.eye(NUM_WEATHER + 1)[..., 1:]
        )
        self.terrain_onehot = nn.Embedding.from_pretrained(
            torch.eye(NUM_TERRAIN + 1)[..., 1:]
        )

        # self.entity_transformer = Transformer(
        #     transformer_num_layers=1,
        #     transformer_num_heads=2,
        #     transformer_key_size=entity_size // 2,
        #     transformer_value_size=entity_size // 2,
        #     transformer_model_size=entity_size,
        #     resblocks_num_before=1,
        #     resblocks_num_after=1,
        #     resblocks_hidden_size=entity_size // 2,
        # )
        self.to_vector = ToVector(entity_size, use_layer_norm=_USE_LAYER_NORM)

        self.turn_embedding = _layer_init(nn.Embedding(64, vector_size))

        boosts_size = NUM_BOOSTS + 12 * NUM_BOOSTS
        self.boosts_embedding = MLP(
            [boosts_size, vector_size], use_layer_norm=_USE_LAYER_NORM
        )

        self.volatile_status_embedding = MLP(
            [NUM_VOLATILE_STATUS, vector_size], use_layer_norm=_USE_LAYER_NORM
        )

        side_condition_size = NUM_SIDE_CONDITIONS + 2 + 3
        self.side_condition_embedding = MLP(
            [side_condition_size, vector_size], use_layer_norm=_USE_LAYER_NORM
        )

        field_size = NUM_PSEUDOWEATHER + NUM_WEATHER + NUM_TERRAIN
        self.field_embedding = MLP(
            [field_size, vector_size], use_layer_norm=_USE_LAYER_NORM
        )

        self.action_rnn = nn.GRU(
            input_size=entity_size,
            hidden_size=vector_size,
            num_layers=1,
            batch_first=True,
        )

        self.side_merge = VectorMerge(
            {
                "side": 2 * entity_size,
                "side_conditions": vector_size,
                "volatile_status": vector_size,
                "boosts": vector_size,
            },
            vector_size,
            gating_type=GatingType.NONE,
            use_layer_norm=_USE_LAYER_NORM,
        )

        self.torso_merge = VectorMerge(
            {
                "myside": vector_size,
                "oppside": vector_size,
                "field": vector_size,
                "action": vector_size,
                "private": 2 * entity_size,
                "turn": vector_size,
            },
            vector_size,
            gating_type=GatingType.NONE,
            use_layer_norm=_USE_LAYER_NORM,
        )

        self.action_stage_embedding = _layer_init(nn.Embedding(3, vector_size))

        self.state_rnn = nn.GRU(
            input_size=vector_size,
            hidden_size=vector_size,
            num_layers=1,
            batch_first=True,
        )

        self.action_type_mlp = MLP(
            [vector_size, vector_size, 2], use_layer_norm=_USE_LAYER_NORM
        )

        self.move_query_mlp = MLP(
            [vector_size, vector_size], use_layer_norm=_USE_LAYER_NORM
        )
        self.move_pointer = PointerLogits(
            vector_size,
            entity_size,
            key_size=entity_size,
            num_layers_keys=1,
            num_layers_query=1,
            use_layer_norm=_USE_LAYER_NORM,
        )

        self.switch_query_mlp = MLP(
            [vector_size, vector_size], use_layer_norm=_USE_LAYER_NORM
        )
        self.switch_pointer = PointerLogits(
            vector_size,
            entity_size,
            key_size=entity_size,
            num_layers_keys=1,
            num_layers_query=1,
            use_layer_norm=_USE_LAYER_NORM,
        )

        self.value_resnet = Resnet(vector_size, use_layer_norm=_USE_LAYER_NORM)
        self.value_mlp = MLP([vector_size, 1], use_layer_norm=_USE_LAYER_NORM)

    def embed_action_history(
        self, entity_embeddings: torch.Tensor, action_history: torch.Tensor
    ):
        T, B, H, *_ = entity_embeddings.shape

        action_hist_mask = action_history[..., 0] >= 0
        action_hist_len = action_hist_mask.sum(-1).flatten().clamp(min=1)
        action_side = action_history[..., 0].clamp(min=0)
        action_user = action_history[..., 1].clamp(min=0)
        action_move = action_history[..., 2] + 2

        indices = F.one_hot(action_user + (action_side == 1) * 6, 18)
        action_user_embeddings = indices.float() @ entity_embeddings.flatten(-3, -2)

        action_embeddings = self.moves_onehot(action_move)
        action_embeddings = action_embeddings * torch.sigmoid(action_user_embeddings)
        action_embeddings = (
            action_embeddings * action_hist_mask.unsqueeze(-1)
        ).flatten(0, -3)

        packed_input = pack_padded_sequence(
            action_embeddings,
            action_hist_len.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, ht = self.action_rnn(packed_input)
        return ht[-1].view(T, B, H, -1)

    def encode_history(
        self, state_embeddings: torch.Tensor, history_mask: torch.Tensor
    ) -> torch.Tensor:
        T, B, H, *_ = state_embeddings.shape
        history_mask_onehot = F.one_hot(history_mask.sum(-1) - 1, H).float()
        output, _ = self.state_rnn(state_embeddings.flatten(0, 1))
        output = output.view(T, B, H, -1)
        final_state = history_mask_onehot.unsqueeze(-2) @ output
        final_state = final_state.view(T, B, -1)
        return output, final_state

    def get_current_entity_embeddings(
        self,
        history_mask: torch.Tensor,
        entity_embeddings: torch.Tensor,
        # active_token: torch.Tensor,
        # ) -> Tuple[torch.Tensor, torch.Tensor]:
    ) -> torch.Tensor:
        T, B, H = history_mask.shape

        my_switch_embeddings = entity_embeddings[:, :, :, 0].transpose(2, -1)
        history_mask_onehot = (
            F.one_hot(history_mask.sum(-1) - 1, H)[..., None, None]
            .transpose(2, -2)
            .detach()
            .float()
        )

        current_ts_embeddings = my_switch_embeddings @ history_mask_onehot
        current_ts_embeddings = current_ts_embeddings.transpose(-3, -1)

        # current_active_token = (
        #     active_token[:, :, :, 0].transpose(2, -1).float() @ history_mask_onehot
        # ).transpose(-2, -1)

        current_ts_embeddings = current_ts_embeddings.view(T, B, 6, -1)
        # current_active_embedding = current_active_token @ current_ts_embeddings
        # current_active_embedding = current_active_embedding.view(T, B, 1, -1)

        return current_ts_embeddings  # , current_active_embedding

    def get_action_mask_context(self, action_mask: torch.Tensor) -> torch.Tensor:
        action_type_select = torch.any(action_mask[..., :2], dim=-1)
        move_select = torch.any(action_mask[..., 2:6], dim=-1)
        switch_select = torch.any(action_mask[..., 6:], dim=-1)
        stage_token = action_type_select * 0 + move_select * 1 + switch_select * 2
        return self.action_stage_embedding(stage_token)

    def embed_teams(self, teams: torch.Tensor) -> torch.Tensor:
        T, B, H, *_ = teams.shape
        teamsp1 = teams + 1
        teamsp2 = teams + 2

        species_token = teamsp1[..., 0]
        species_onehot = self.species_onehot(species_token)

        item_token = teamsp1[..., 1]
        item_onehot = self.item_onehot(item_token)

        ability_token = teamsp1[..., 2]
        ability_onehot = self.ability_onehot(ability_token)
        # hp_value = teams[..., 3, None].float() / 1024
        hp_bucket = torch.sqrt(teams[..., 3]).to(torch.long)
        hp_onehot = self.hp_onehot(hp_bucket)

        active_token = teams[..., 4]
        active_onehot = self.active_onehot(active_token)

        fainted_token = teams[..., 5]
        fainted_onehot = self.fainted_onehot(fainted_token)

        status_token = teamsp1[..., 6]
        status_onehot = self.status_onehot(status_token)

        move_tokens = teamsp2[..., -4:]
        moveset_onehot = self.moves_onehot(move_tokens).sum(-2)
        # public_onehot = self.public_onehot(
        #     self.public_token[(None,) * 3].expand(T, B, H, 3, 6)
        # )

        side_token = self.side_token[(None,) * 3].expand(T, B, H, 3, 6)
        side_onehot = self.side_onehot(side_token)

        encodings = (
            species_onehot
            + item_onehot
            + ability_onehot
            + hp_onehot
            # + hp_value
            + active_onehot
            + fainted_onehot
            + status_onehot
            + moveset_onehot
            # + public_onehot
            + side_onehot
        )
        encodings = self.units_mlp(encodings)
        encodings, loss = self.unit_quantizer(encodings)

        return encodings, loss

    def embed_side_conditions(
        self, side_conditions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        other = side_conditions > 0
        spikes = self.spikes_onehot(side_conditions[..., 9])
        tspikes = self.tspikes_onehot(side_conditions[..., 13])
        encoding = torch.cat((other, spikes, tspikes), -1)
        return torch.chunk(self.side_condition_embedding(encoding), 2, -2)

    def embed_volatile_status(
        self, volatile_status: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        volatile_status_id = volatile_status[..., 0]
        volatile_status_level = volatile_status[..., 1]
        encoding = self.volatile_status_onehot(volatile_status_id + 1).sum(-2)
        return torch.chunk(self.volatile_status_embedding(encoding), 2, -2)

    def embed_boosts(self, boosts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        boosts_onehot = self.boosts_onehot(boosts + 6)
        boosts_onehot = torch.cat((boosts_onehot[..., :6], boosts_onehot[..., 7:]), -1)
        boosts_scaled = torch.sign(boosts) * torch.sqrt(abs(boosts))
        encoding = torch.cat((boosts_onehot.flatten(-2), boosts_scaled), -1)
        return torch.chunk(self.boosts_embedding(encoding), 2, -2)

    def softmax_passthrough(self, tensor: torch.Tensor, buckets: int = 16):
        og_shape = tensor.shape
        flat_tensor = tensor.reshape(-1, buckets)
        gsm = F.gumbel_softmax(flat_tensor)
        return gsm.view(*og_shape)

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
        return self.field_embedding(encoding)

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
        history_mask: torch.Tensor,
    ):
        entity_embeddings, recon_loss = self.embed_teams(teams)
        recon_loss = (recon_loss * history_mask).sum(-1) / history_mask.sum(-1)
        recon_loss = recon_loss / NUM_HISTORY

        # entity_embeddings_attn = self.entity_transformer(
        #     entity_embeddings.flatten(-3, -2),
        #     torch.ones_like(
        #         entity_embeddings.flatten(-3, -2)[..., 0].squeeze(-1), dtype=torch.bool
        #     ),
        # )
        # entity_embeddings = entity_embeddings_attn.view_as(entity_embeddings)

        active_token = teams[..., 4]
        (
            my_private_side_embedding,
            my_public_side_embedding,
            opp_side_embedding,
        ) = self.to_vector(entity_embeddings, active_token)

        private_information = my_private_side_embedding - my_public_side_embedding

        (
            my_side_conditions_embedding,
            opp_side_conditions_embedding,
        ) = self.embed_side_conditions(side_conditions)
        (
            my_volatile_status_embedding,
            opp_volatile_status_embedding,
        ) = self.embed_volatile_status(volatile_status)
        my_boosts_embedding, opp_boosts_embedding = self.embed_boosts(boosts)

        my_side_embedding = self.side_merge(
            {
                "side": my_private_side_embedding.squeeze(-2),
                "side_conditions": my_side_conditions_embedding.squeeze(-2),
                "volatile_status": my_volatile_status_embedding.squeeze(-2),
                "boosts": my_boosts_embedding.squeeze(-2),
            }
        )
        opp_side_embedding = self.side_merge(
            {
                "side": opp_side_embedding.squeeze(-2),
                "side_conditions": opp_side_conditions_embedding.squeeze(-2),
                "volatile_status": opp_volatile_status_embedding.squeeze(-2),
                "boosts": opp_boosts_embedding.squeeze(-2),
            }
        )

        field_embedding = self.embed_field(field)

        prev_entity_embeddings = torch.cat(
            (entity_embeddings[:, :, :1], entity_embeddings[:, :, :-1]), dim=2
        )
        action_hist_embedding = self.embed_action_history(
            prev_entity_embeddings, history
        )
        turn_embedding = self.turn_embedding(
            turn.clamp(min=0, max=self.turn_embedding.weight.shape[0] - 1)
        )

        step_embeddings = self.torso_merge(
            {
                "myside": my_side_embedding,
                "oppside": opp_side_embedding,
                "field": field_embedding,
                "action": action_hist_embedding,
                "private": private_information.squeeze(-2),
                "turn": turn_embedding,
            }
        )

        step_embeddings, state_embedding = self.encode_history(
            step_embeddings, history_mask
        )

        state_embedding_w_context = state_embedding + self.get_action_mask_context(
            legal
        )
        # state_embedding_w_context = self.softmax_passthrough(state_embedding_w_context)

        active_token = teams[..., 4]
        switch_embeddings = self.get_current_entity_embeddings(
            history_mask, entity_embeddings  # , active_token
        )
        # switch_embeddings = self.softmax_passthrough(switch_embeddings)

        move_embeddings = self.moves_onehot(active_moveset)
        move_embeddings = self.moves_mlp(move_embeddings)
        # move_embeddings = self.softmax_passthrough(move_embeddings)

        action_type_logits = self.action_type_mlp(state_embedding_w_context)

        move_query = self.move_query_mlp(state_embedding_w_context).unsqueeze(-2)
        move_logits = self.move_pointer(move_query, move_embeddings).flatten(2)

        switch_query = self.switch_query_mlp(state_embedding_w_context).unsqueeze(-2)
        switch_logits = self.switch_pointer(switch_query, switch_embeddings).flatten(2)

        logits = torch.cat((action_type_logits, move_logits, switch_logits), dim=-1)

        policy = _legal_policy(logits, legal)
        log_policy = _legal_log_policy(logits, legal)

        value = self.value_mlp(self.value_resnet(state_embedding_w_context))

        return ModelOutput(
            logits=logits,
            policy=policy,
            log_policy=log_policy,
            value=value,
            recon_loss=recon_loss,
        )
