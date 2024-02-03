import torch
import torch.nn as nn

from pokesim.data import (
    NUM_ABILITIES,
    NUM_BOOSTS,
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

from typing import Tuple

from pokesim.structs import ModelOutput
from pokesim.utils import _legal_log_policy, _legal_policy

from pokesim.nn.helpers import _layer_init
from pokesim.nn.modules import (
    GatingType,
    MultiHeadAttention,
    PointerLogits,
    Transformer,
    VectorMerge,
    MLP,
)

# from pokemb.mod import ComponentEmbedding


PADDING_TOKEN = -1
UNKNOWN_TOKEN = -2
SWITCH_TOKEN = -3


class Encoder(nn.Module):
    def __init__(
        self,
        entity_size: int,
        stream_size: int,
        use_layer_norm: bool,
        affine_layer_norm: bool,
    ):
        super().__init__()

        self.species_onehot = nn.Embedding.from_pretrained(torch.eye(NUM_SPECIES + 2))
        self.item_onehot = nn.Embedding.from_pretrained(torch.eye(NUM_ITEMS + 2))
        self.ability_onehot = nn.Embedding.from_pretrained(torch.eye(NUM_ABILITIES + 2))
        self.moves_onehot = nn.Embedding.from_pretrained(torch.eye(NUM_MOVES + 3))
        self.hp_onehot = nn.Embedding.from_pretrained(torch.eye(65))
        self.status_onehot = nn.Embedding.from_pretrained(torch.eye(NUM_STATUS + 1))
        self.active_onehot = nn.Embedding.from_pretrained(torch.eye(2))
        self.fainted_onehot = nn.Embedding.from_pretrained(torch.eye(2))
        self.side_onehot = nn.Embedding.from_pretrained(torch.eye(2))
        self.public_onehot = nn.Embedding.from_pretrained(torch.eye(2))
        self.sleep_turns_onehot = nn.Embedding.from_pretrained(torch.eye(4))
        self.toxic_turns_onehot = nn.Embedding.from_pretrained(torch.eye(6))

        self.entity_cls = _layer_init(nn.Parameter(torch.randn(1, 1, 4, entity_size)))
        self.history_cls = _layer_init(nn.Parameter(torch.randn(1, 1, 4, entity_size)))
        self.entity_lin = nn.Sequential(
            _layer_init(
                nn.Linear(
                    self.species_onehot.weight.shape[-1]
                    + self.item_onehot.weight.shape[-1]
                    + self.ability_onehot.weight.shape[-1]
                    + 2 * self.moves_onehot.weight.shape[-1]
                    + 1 * self.hp_onehot.weight.shape[-1]
                    + self.status_onehot.weight.shape[-1]
                    + self.active_onehot.weight.shape[-1]
                    + self.fainted_onehot.weight.shape[-1]
                    + self.side_onehot.weight.shape[-1]
                    + self.public_onehot.weight.shape[-1]
                    + self.sleep_turns_onehot.weight.shape[-1]
                    + self.toxic_turns_onehot.weight.shape[-1],
                    entity_size,
                )
            ),
            MLP([entity_size, entity_size, entity_size]),
        )

        self.is_critical_onehot = nn.Embedding.from_pretrained(torch.eye(2))
        self.effectiveness_onehot = nn.Embedding.from_pretrained(torch.eye(5))
        self.missed_onehot = nn.Embedding.from_pretrained(torch.eye(2))
        self.target_fainted_onehot = nn.Embedding.from_pretrained(torch.eye(2))
        self.damage_value_onehot = nn.Embedding.from_pretrained(torch.eye(64))
        self.action_counter_onehot = nn.Embedding.from_pretrained(torch.eye(4))
        self.order_onehot = nn.Embedding.from_pretrained(torch.eye(20))

        self.boosts_onehot = nn.Embedding.from_pretrained(torch.eye(13))

        self.spikes_onehot = nn.Embedding.from_pretrained(torch.eye(4)[..., 1:])
        self.tspikes_onehot = nn.Embedding.from_pretrained(torch.eye(3)[..., 1:])
        self.volatile_status_onehot = nn.Embedding.from_pretrained(
            torch.eye(NUM_VOLATILE_STATUS + 1)[..., 1:]
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

        context_size = (
            2 * (12 * NUM_BOOSTS + NUM_VOLATILE_STATUS + NUM_SIDE_CONDITIONS + 2 + 3)
            + field_size
        )
        stats_size = (
            self.is_critical_onehot.weight.shape[-1]
            + self.effectiveness_onehot.weight.shape[-1]
            + self.missed_onehot.weight.shape[-1]
            + self.target_fainted_onehot.weight.shape[-1]
            + self.damage_value_onehot.weight.shape[-1]
            + 1
            + 2 * self.action_counter_onehot.weight.shape[-1]
            + self.moves_onehot.weight.shape[-1]
            + self.order_onehot.weight.shape[-1]
        )

        self.history_diff_lin = _layer_init(nn.Linear(3 * entity_size, entity_size))
        self.history_prev_lin = _layer_init(nn.Linear(3 * entity_size, entity_size))
        self.history_stats_lin = _layer_init(nn.Linear(stats_size, entity_size))

        self.history_merge = VectorMerge(
            input_sizes={
                "user": entity_size,
                "target": entity_size,
                "context": entity_size,
                "stats": entity_size,
            },
            output_size=entity_size,
            gating_type=GatingType.NONE,
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )
        self.history_mlp = MLP([entity_size, entity_size])

        self.transformer = Transformer(
            units_stream_size=entity_size,
            transformer_num_layers=1,
            transformer_num_heads=2,
            transformer_key_size=entity_size // 2,
            transformer_value_size=entity_size // 2,
            resblocks_num_before=0,
            resblocks_num_after=1,
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )

        self.context_lin = nn.Sequential(
            _layer_init(nn.Linear(context_size, 2 * entity_size)),
        )
        self.context_mlp = MLP([2 * entity_size, stream_size])
        self.history_context_mlp = MLP([2 * entity_size, entity_size])

    def forward_entities(self, entities: torch.Tensor) -> torch.Tensor:
        entitiesp1 = entities + 1
        entitiesp2 = entities + 2
        entitiesp3 = entities + 3

        species_token = entitiesp2[..., 0]
        item_token = entitiesp2[..., 1]
        ability_token = entitiesp2[..., 2]
        hp = entities[..., 3].clamp(0, 1024)
        hp_bucket = torch.floor(hp * 64 / 1024).to(torch.long)
        active_token = entities[..., 4]
        fainted_token = entities[..., 5]
        status_token = entitiesp1[..., 6]
        last_move_token = entitiesp3[..., 7]
        public_token = entities[..., 8]
        side_token = entities[..., 9]
        sleep_turns_token = entities[..., 10].clamp(min=0, max=3)
        toxic_turns_token = entities[..., 11].clamp(min=0, max=5)
        move_pp_left = entities[..., 12:16]
        move_pp_max = entities[..., 16:20]
        move_tokens = entitiesp3[..., 20:]

        species_onehot = self.species_onehot(species_token)
        item_onehot = self.item_onehot(item_token)
        ability_onehot = self.ability_onehot(ability_token)
        hp_onehot = self.hp_onehot(hp_bucket)
        active_onehot = self.active_onehot(active_token)
        fainted_onehot = self.fainted_onehot(fainted_token)
        status_onehot = self.status_onehot(status_token)
        last_move_onehot = self.moves_onehot(last_move_token)
        side_onehot = self.side_onehot(side_token)
        public_onehot = self.public_onehot(public_token)
        sleep_turns_onehot = self.sleep_turns_onehot(sleep_turns_token)
        toxic_turns_onehot = self.toxic_turns_onehot(toxic_turns_token)
        moveset_onehot = (self.moves_onehot(move_tokens) / 4).sum(-2)

        onehot = torch.cat(
            (
                species_onehot,
                item_onehot,
                ability_onehot,
                last_move_onehot,
                hp_onehot,
                # prev_hp_onehot,
                active_onehot,
                fainted_onehot,
                status_onehot,
                side_onehot,
                public_onehot,
                sleep_turns_onehot,
                toxic_turns_onehot,
                moveset_onehot,
            ),
            dim=-1,
        )
        raw_embeddings = self.entity_lin(onehot)

        return raw_embeddings

    def concat_cls_entities(self, embeddings: torch.Tensor):
        expanded_cls = self.entity_cls.expand(*embeddings.shape[:2], -1, -1)
        return torch.cat((expanded_cls, embeddings), dim=-2)

    def forward_history_stats(self, stats: torch.Tensor) -> torch.Tensor:
        statsp3 = stats + 3

        is_critical_token = stats[..., 0].clamp(min=0)
        effectiveness_token = stats[..., 1].clamp(min=0)
        missed_token = stats[..., 2].clamp(min=0)
        target_fainted_token = stats[..., 3].clamp(min=0)
        damage_value_scalar = (stats[..., 4].unsqueeze(-1) / 2047) - 1
        damage_value_token = (
            (stats[..., 4].clamp(min=0) * 63 / 4095).floor().to(torch.long)
        )
        move_counter = stats[..., 5].clamp(min=0, max=3)
        switch_counter = stats[..., 6].clamp(min=0, max=3)
        move_token = statsp3[..., 7]
        order = stats[..., 8].max(-1, keepdim=True).values - stats[..., 8]

        is_critical_onehot = self.is_critical_onehot(is_critical_token)
        effectivesness_onehot = self.effectiveness_onehot(effectiveness_token)
        missed_onehot = self.missed_onehot(missed_token)
        target_fainted_onehot = self.target_fainted_onehot(target_fainted_token)
        damage_value_onehot = self.damage_value_onehot(damage_value_token)
        move_counter_onehot = self.action_counter_onehot(move_counter)
        switch_counter_onehot = self.action_counter_onehot(switch_counter)
        move_token_onehot = self.moves_onehot(move_token)
        order_onehot = self.order_onehot(
            order.clamp(min=0, max=self.order_onehot.weight.shape[-1] - 1)
        )

        onehot = torch.cat(
            (
                is_critical_onehot,
                effectivesness_onehot,
                missed_onehot,
                target_fainted_onehot,
                damage_value_scalar,
                damage_value_onehot,
                move_counter_onehot,
                switch_counter_onehot,
                move_token_onehot,
                order_onehot,
            ),
            dim=-1,
        )

        return self.history_stats_lin(onehot)

    def encode_side_conditions(self, side_conditions: torch.Tensor) -> torch.Tensor:
        spikes_token = side_conditions[..., 9].clamp(
            min=0, max=self.spikes_onehot.weight.shape[-1] - 1
        )
        tspikes_token = side_conditions[..., 13].clamp(
            min=0, max=self.tspikes_onehot.weight.shape[-1] - 1
        )
        other = side_conditions > 0
        spikes = self.spikes_onehot(spikes_token)
        tspikes = self.tspikes_onehot(tspikes_token)
        return torch.cat((other, spikes, tspikes), -1)

    def encode_volatile_status(self, volatile_status: torch.Tensor) -> torch.Tensor:
        volatile_status_id = volatile_status[..., 0]
        volatile_status_level = volatile_status[..., 1]
        return self.volatile_status_onehot(volatile_status_id + 1).sum(-2)

    def encode_boosts(self, boosts: torch.Tensor) -> torch.Tensor:
        boost_token = (boosts + 6).clamp(
            min=0, max=self.boosts_onehot.weight.shape[-1] - 1
        )
        boosts_onehot = self.boosts_onehot(boost_token)
        return torch.cat((boosts_onehot[..., :6], boosts_onehot[..., 7:]), -1)

    def encode_field(self, field: torch.Tensor) -> torch.Tensor:
        field_id = field[..., 0]
        field_id_p1 = field_id + 1

        field_min_durr = field[..., 1]
        field_max_durr = field[..., 2]

        pseudoweathers = field_id_p1[..., :3].clamp(
            min=0, max=self.pseudoweathers_onehot.weight.shape[-1] - 1
        )
        weather = field_id_p1[..., 3].clamp(
            min=0, max=self.weathers_onehot.weight.shape[-1] - 1
        )
        terrain = field_id_p1[..., 4].clamp(
            min=0, max=self.terrain_onehot.weight.shape[-1] - 1
        )

        pseudoweathers_onehot = self.pseudoweathers_onehot(pseudoweathers).sum(-2)
        weather_onehot = self.weathers_onehot(weather)
        terrain_onehot = self.terrain_onehot(terrain)

        return torch.cat((pseudoweathers_onehot, weather_onehot, terrain_onehot), -1)

    def embed_context(
        self,
        side_conditions: torch.Tensor,
        volatile_status: torch.Tensor,
        boosts: torch.Tensor,
        field: torch.Tensor,
    ) -> torch.Tensor:
        side_conditions_encoding = self.encode_side_conditions(side_conditions)
        volatile_status_encoding = self.encode_volatile_status(volatile_status)
        boosts_encoding = self.encode_boosts(boosts)
        field_encoding = self.encode_field(field)
        context_encoding = torch.cat(
            (
                side_conditions_encoding.flatten(-2),
                volatile_status_encoding.flatten(-2),
                boosts_encoding.flatten(-3),
                field_encoding,
            ),
            dim=-1,
        )
        return self.context_lin(context_encoding)

    def forward_history(
        self,
        history_user_embeddings: torch.Tensor,
        history_target_embeddings: torch.Tensor,
        history_context_embedding: torch.Tensor,
        history_stats_embedding: torch.Tensor,
    ) -> torch.Tensor:
        history_embeddings = self.history_merge(
            {
                "user": history_user_embeddings,
                "target": history_target_embeddings,
                "context": history_context_embedding,
                "stats": history_stats_embedding,
            },
        )

        history_embeddings = torch.cat(
            (
                self.history_cls.expand(*history_embeddings.shape[:2], -1, -1),
                history_embeddings,
            ),
            dim=-2,
        )
        return self.history_mlp(history_embeddings)

    def get_history_mask(self, history_stats: torch.Tensor) -> torch.Tensor:
        history_mask = history_stats[:, :, -1, ..., 0].to(torch.bool)
        return torch.cat((torch.ones_like(history_mask[..., :4]), history_mask), dim=-1)

    def forward(
        self,
        teams: torch.Tensor,
        side_conditions: torch.Tensor,
        volatile_status: torch.Tensor,
        boosts: torch.Tensor,
        field: torch.Tensor,
        history_side_conditions: torch.Tensor,
        history_volatile_status: torch.Tensor,
        history_boosts: torch.Tensor,
        history_field: torch.Tensor,
        history_entities: torch.Tensor,
        history_stats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # mask = teams[..., -1, :, :, 0] != PADDING_TOKEN

        raw_embeddings = self.forward_entities(teams[:, :, -1]).flatten(2, 3)
        entity_embeddings = self.concat_cls_entities(raw_embeddings)

        entity_mask = torch.ones_like(entity_embeddings[..., 0], dtype=torch.bool)

        active_token = teams[..., -1, :, :, 4]

        history_context_embeddings = self.embed_context(
            history_side_conditions[:, :, -1],
            history_volatile_status[:, :, -1],
            history_boosts[:, :, -1],
            history_field[:, :, -1],
        )
        history_context_embeddings = self.history_context_mlp(
            history_context_embeddings
        )
        history_entity_embeddings = self.forward_entities(history_entities[:, :, -1])
        history_user = history_entity_embeddings[..., 0, :]
        history_target = history_entity_embeddings[..., 1, :]

        history_stats_embedding = self.forward_history_stats(
            history_stats[:, :, -1, ..., 1:]
        )

        history_embeddings = self.forward_history(
            history_user,
            history_target,
            history_context_embeddings,
            history_stats_embedding,
        )

        history_mask = self.get_history_mask(history_stats)

        entity_embeddings = self.transformer(
            entity_embeddings, history_embeddings, entity_mask, history_mask
        )

        active_weight = active_token[..., 0, :].unsqueeze(-2).float()
        active_embedding = active_weight @ entity_embeddings[..., 4:10, :]

        entities_embedding = entity_embeddings[..., :4, :].flatten(2)
        entity_embeddings = entity_embeddings[..., 4:10, :]

        context_encoding = self.embed_context(
            side_conditions[:, :, -1],
            volatile_status[:, :, -1],
            boosts[:, :, -1],
            field[:, :, -1],
        )
        context_embedding = self.context_mlp(context_encoding)

        return (
            active_weight,
            active_embedding,
            entity_embeddings,
            entities_embedding,
            context_embedding,
        )


class Torso(nn.Module):
    def __init__(
        self,
        stream_size: int,
        use_layer_norm: bool,
        affine_layer_norm: bool,
    ):
        super().__init__()

        self.torso_merge = VectorMerge(
            input_sizes={"entities": stream_size, "context": stream_size},
            output_size=stream_size,
            gating_type=GatingType.NONE,
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )
        # self.torso_resnet = Resnet(
        #     input_size=stream_size,
        #     num_resblocks=2,
        #     use_layer_norm=use_layer_norm,
        #     affine_layer_norm=affine_layer_norm,
        # )

    def forward(
        self, entities_embedding: torch.Tensor, context_embedding: torch.Tensor
    ) -> torch.Tensor:
        state_embedding = self.torso_merge(
            {"entities": entities_embedding, "context": context_embedding}
        )
        # state_embedding = self.torso_resnet(state_embedding)
        return state_embedding


class PolicyHead(nn.Module):
    def __init__(
        self,
        entity_size: int,
        stream_size: int,
        use_layer_norm: bool,
        affine_layer_norm: bool,
    ):
        super().__init__()

        switch_tokens = SWITCH_TOKEN * torch.ones(6, dtype=torch.long)
        self.register_buffer("switch_tokens", switch_tokens)

        self.moves_onehot = nn.Embedding.from_pretrained(torch.eye(NUM_MOVES + 3))
        self.legal_onehot = nn.Embedding.from_pretrained(torch.eye(2))

        self.legal_embedding = nn.Embedding.from_pretrained(torch.eye(2))

        self.actions_lin = _layer_init(
            nn.Linear(
                self.moves_onehot.weight.shape[-1]
                + self.legal_embedding.weight.shape[-1],
                entity_size,
            )
        )

        self.action_merge = VectorMerge(
            {"action": entity_size, "user": entity_size},
            output_size=entity_size,
            gating_type=GatingType.NONE,
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )

        self.action_self_attn = MultiHeadAttention(
            num_heads=2,
            key_size=entity_size // 2,
            value_size=entity_size // 2,
            model_size=entity_size,
        )

        self.action_logits = PointerLogits(
            stream_size,
            entity_size,
            num_layers_query=1,
            num_layers_keys=3,
            key_size=entity_size,
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )

    def get_action_tokens(
        self, active_weight: torch.Tensor, teams: torch.Tensor
    ) -> torch.Tensor:
        return (
            (active_weight @ teams.flatten(3, 4)[:, :, -1, :6].float())[..., -12:]
            .squeeze(-2)
            .long()
        )

    def embed_actions(
        self, active_movesets: torch.Tensor, legal: torch.Tensor
    ) -> torch.Tensor:
        moveset_pp_left = active_movesets[..., :4]
        moveset_pp_max = active_movesets[..., 4:8]
        moveset_tokens = active_movesets[..., 8:]

        expanded_switch_tokens = self.switch_tokens.view(1, 1, -1).expand(
            *active_movesets.shape[:2], 6
        )
        action_tokens = torch.cat((moveset_tokens, expanded_switch_tokens), dim=-1)
        action_tokens = action_tokens - SWITCH_TOKEN

        actions_onehot = torch.cat(
            (
                self.moves_onehot(action_tokens),
                self.legal_onehot(legal.to(torch.long)),
            ),
            dim=-1,
        )

        return self.actions_lin(actions_onehot)

    def forward(
        self,
        active_weight: torch.Tensor,
        teams: torch.Tensor,
        legal: torch.Tensor,
        active_embedding: torch.Tensor,
        entity_embeddings: torch.Tensor,
        state_embedding: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        action_tokens = self.get_action_tokens(active_weight, teams)
        action_embeddings = self.embed_actions(action_tokens, legal)

        user_embeddings = torch.cat(
            (active_embedding.expand(-1, -1, 4, -1), entity_embeddings), dim=-2
        )

        # action_query = self.action_query_resnet(action_query)
        action_embeddings = self.action_merge(
            {"action": action_embeddings, "user": user_embeddings}
        )
        # action_embeddings = self.action_keys_resnet(action_embeddings)
        action_embeddings = self.action_self_attn(
            action_embeddings, action_embeddings, action_embeddings
        )

        # logits = self.action_logits(action_embeddings).flatten(2)
        logits = self.action_logits(
            state_embedding.unsqueeze(-2), action_embeddings
        ).flatten(2)

        policy = _legal_policy(logits, legal)
        log_policy = _legal_log_policy(logits, legal)

        return policy, log_policy, logits


class ValueHead(nn.Module):
    def __init__(
        self,
        stream_size: int,
        use_layer_norm: bool,
        affine_layer_norm: bool,
    ):
        super().__init__()

        self.value_mlp = MLP(
            [stream_size, stream_size, 1],
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )

    def forward(self, state_embedding: torch.Tensor) -> torch.Tensor:
        value_hidden = state_embedding
        # value_hidden = self.value_resnet(state_embedding)
        value = self.value_mlp(value_hidden)
        return value


class Model(nn.Module):
    def __init__(
        self,
        entity_size: int = 32,
        stream_size: int = 128,
        scale: int = 8,
        use_layer_norm: bool = True,
        affine_layer_norm: bool = False,
    ):
        super().__init__()

        entity_size *= scale
        stream_size *= scale

        self.entity_size = entity_size
        self.stream_size = stream_size

        self.encoder = Encoder(
            entity_size=entity_size,
            stream_size=stream_size,
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )

        self.torso = Torso(
            stream_size=stream_size,
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )

        self.policy_head = PolicyHead(
            entity_size=entity_size,
            stream_size=stream_size,
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )

        self.value_head = ValueHead(
            stream_size=stream_size,
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )

    def forward(
        self,
        turn: torch.Tensor,
        teams: torch.Tensor,
        side_conditions: torch.Tensor,
        volatile_status: torch.Tensor,
        boosts: torch.Tensor,
        field: torch.Tensor,
        legal: torch.Tensor,
        history_side_conditions: torch.Tensor,
        history_volatile_status: torch.Tensor,
        history_boosts: torch.Tensor,
        history_field: torch.Tensor,
        history_entities: torch.Tensor,
        history_stats: torch.Tensor,
    ):
        (
            active_weight,
            active_embedding,
            entity_embeddings,
            entities_embedding,
            context_embedding,
        ) = self.encoder.forward(
            teams=teams,
            side_conditions=side_conditions,
            volatile_status=volatile_status,
            boosts=boosts,
            field=field,
            history_side_conditions=history_side_conditions,
            history_volatile_status=history_volatile_status,
            history_boosts=history_boosts,
            history_field=history_field,
            history_entities=history_entities,
            history_stats=history_stats,
        )

        state_embedding = self.torso.forward(
            entities_embedding=entities_embedding,
            context_embedding=context_embedding,
        )

        policy, log_policy, logits = self.policy_head.forward(
            active_weight=active_weight,
            teams=teams,
            legal=legal,
            active_embedding=active_embedding,
            entity_embeddings=entity_embeddings,
            state_embedding=state_embedding,
        )

        value = self.value_head.forward(state_embedding=state_embedding)

        return ModelOutput(
            policy=policy,
            log_policy=log_policy,
            logits=logits,
            value=value,
        )
