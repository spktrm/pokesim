import torch
import torch.nn as nn

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
    CrossTransformer,
    GatingType,
    PointerLogits,
    # Resnet,
    # ToVector,
    Transformer,
    VectorMerge,
    MLP,
)

# from pokemb.mod import ComponentEmbedding


PADDING_TOKEN = -1
UNKNOWN_TOKEN = -2
SWITCH_TOKEN = -3


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

        side_tokens = torch.tensor([0] * 12 + [1] * 6, dtype=torch.long).view(-1, 6)
        self.register_buffer("side_tokens", side_tokens[None, None])

        public_tokens = torch.tensor([0] * 6 + [1] * 12, dtype=torch.long).view(-1, 6)
        self.register_buffer("public_tokens", public_tokens[None, None])

        switch_tokens = SWITCH_TOKEN * torch.ones(6, dtype=torch.long)
        self.register_buffer("switch_tokens", switch_tokens)

        # self.species_embedding = ComponentEmbedding(
        #     name="species", output_size=entity_size, gen=3, num_unknown=2
        # )

        self.species_onehot = nn.Embedding.from_pretrained(torch.eye(NUM_SPECIES + 2))
        self.item_onehot = nn.Embedding.from_pretrained(torch.eye(NUM_ITEMS + 2))
        self.ability_onehot = nn.Embedding.from_pretrained(torch.eye(NUM_ABILITIES + 2))
        self.moves_onehot = nn.Embedding.from_pretrained(torch.eye(NUM_MOVES + 3))
        self.hp_onehot = nn.Embedding.from_pretrained(torch.eye(NUM_HP_BUCKETS + 1))
        self.status_onehot = nn.Embedding.from_pretrained(torch.eye(NUM_STATUS + 1))
        self.active_onehot = nn.Embedding.from_pretrained(torch.eye(2))
        self.fainted_onehot = nn.Embedding.from_pretrained(torch.eye(2))
        self.side_onehot = nn.Embedding.from_pretrained(torch.eye(2))
        self.public_onehot = nn.Embedding.from_pretrained(torch.eye(2))

        self.entity_cls = _layer_init(nn.Parameter(torch.randn(1, 1, 4, entity_size)))
        self.entity_lin = _layer_init(
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
                + self.public_onehot.weight.shape[-1],
                entity_size,
            )
        )

        self.entity_transformer = Transformer(
            units_stream_size=entity_size,
            transformer_num_layers=3,
            transformer_num_heads=2,
            transformer_key_size=entity_size // 2,
            transformer_value_size=entity_size // 2,
            resblocks_num_before=2,
            resblocks_num_after=2,
            resblocks_hidden_size=entity_size // 2,
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )
        self.entity_moves_transformer = CrossTransformer(
            units_stream_size=entity_size,
            transformer_num_heads=2,
            transformer_key_size=entity_size // 2,
            transformer_value_size=entity_size // 2,
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )

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

        self.context_linear = _layer_init(
            nn.Linear(
                2
                * (12 * NUM_BOOSTS + NUM_VOLATILE_STATUS + NUM_SIDE_CONDITIONS + 2 + 3)
                + field_size,
                stream_size,
            )
        )
        self.entities_to_vector = MLP(
            [3 * entity_size, stream_size],
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )

        self.torso_merge = VectorMerge(
            input_sizes={"entities": stream_size, "context": stream_size},
            output_size=stream_size,
            gating_type=GatingType.POINTWISE,
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )
        # self.torso_resnet = Resnet(
        #     input_size=stream_size,
        #     num_resblocks=2,
        #     use_layer_norm=use_layer_norm,
        #     affine_layer_norm=affine_layer_norm,
        # )

        self.action_merge = VectorMerge(
            {
                # "query": stream_size,
                "action": self.moves_onehot.weight.shape[-1],
                "user": entity_size,
            },
            output_size=entity_size,
            gating_type=GatingType.NONE,
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
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

        self.value_mlp = MLP(
            [stream_size, stream_size, 1],
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )

    def embed_teams(self, teams: torch.Tensor) -> torch.Tensor:
        teamsp1 = teams + 1
        teamsp2 = teams + 2
        teamsp3 = teams + 3

        species_token = teamsp2[..., 0]
        item_token = teamsp2[..., 1]
        ability_token = teamsp2[..., 2]
        hp = teams[..., 3]
        hp_bucket = torch.sqrt(hp).to(torch.long)
        active_token = teams[..., 4]
        fainted_token = teams[..., 5]
        status_token = teamsp1[..., 6]
        last_move_token = teamsp3[..., 7]
        # prev_hp = teams[..., 8]
        # prev_hp_bucket = torch.sqrt(prev_hp).to(torch.long)
        move_tokens = teamsp3[..., -8:-4]
        move_pps = teamsp3[..., -4:]

        side_token = self.side_tokens.expand(*teams.shape[:2], -1, -1)
        public_token = self.public_tokens.expand(*teams.shape[:2], -1, -1)

        species_onehot = self.species_onehot(species_token)
        item_onehot = self.item_onehot(item_token)
        ability_onehot = self.ability_onehot(ability_token)
        hp_onehot = self.hp_onehot(hp_bucket)
        # prev_hp_onehot = self.hp_onehot(prev_hp_bucket)
        active_onehot = self.active_onehot(active_token)
        fainted_onehot = self.fainted_onehot(fainted_token)
        status_onehot = self.status_onehot(status_token)
        side_onehot = self.side_onehot(side_token)
        public_onehot = self.public_onehot(public_token)
        last_move_onehot = self.moves_onehot(last_move_token)
        moveset_onehot = (self.moves_onehot(move_tokens) / 4).sum(-2)

        # hp_value = teams[..., 3, None].float() / 1024

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
                moveset_onehot,
            ),
            dim=-1,
        )
        embeddings = self.entity_lin(onehot).flatten(2, 3)
        embeddings = torch.cat(
            (self.entity_cls.expand(*embeddings.shape[:2], -1, -1), embeddings), dim=-2
        )

        return embeddings

    def encode_side_conditions(self, side_conditions: torch.Tensor) -> torch.Tensor:
        other = side_conditions > 0
        spikes = self.spikes_onehot(side_conditions[..., 9])
        tspikes = self.tspikes_onehot(side_conditions[..., 13])
        return torch.cat((other, spikes, tspikes), -1)

    def encode_volatile_status(self, volatile_status: torch.Tensor) -> torch.Tensor:
        volatile_status_id = volatile_status[..., 0]
        volatile_status_level = volatile_status[..., 1]
        return self.volatile_status_onehot(volatile_status_id + 1).sum(-2)

    def encode_boosts(self, boosts: torch.Tensor) -> torch.Tensor:
        boosts_onehot = self.boosts_onehot(boosts + 6)
        return torch.cat((boosts_onehot[..., :6], boosts_onehot[..., 7:]), -1)

    def encode_field(self, field: torch.Tensor) -> torch.Tensor:
        field_id = field[..., 0]
        field_min_durr = field[..., 1]
        field_max_durr = field[..., 2]

        pseudoweathers = field_id[..., :3]
        pseudoweathers_onehot = self.pseudoweathers_onehot(pseudoweathers + 1).sum(-2)

        weather = field_id[..., 3]
        weather_onehot = self.weathers_onehot(weather + 1)

        terrain = field_id[..., 4]
        terrain_onehot = self.terrain_onehot(terrain + 1)

        return torch.cat((pseudoweathers_onehot, weather_onehot, terrain_onehot), -1)

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
        history_side_conditions: torch.Tensor,
        history_volatile_status: torch.Tensor,
        history_boosts: torch.Tensor,
        history_field: torch.Tensor,
        history_entities: torch.Tensor,
        history_stats: torch.Tensor,
    ):
        entity_embeddings = self.embed_teams(teams[:, :, -1])
        mask = teams[..., -1, :, :, 0] != PADDING_TOKEN
        history_mask = history_stats[:, :, -1, ..., -1] != -1
        history_mask = history_mask.view(*history_mask.shape, 1, 1)

        active_token = teams[..., -1, :, :, 4]
        entity_embeddings = self.entity_transformer(
            entity_embeddings, mask.flatten(2, 3).bool()
        )

        history_side_conditions_encoding = self.encode_side_conditions(
            history_side_conditions[:, :, -1] * history_mask
        )
        history_volatile_status_encoding = self.encode_volatile_status(
            history_volatile_status[:, :, -1] * history_mask.unsqueeze(-1)
        )
        history_boosts_encoding = self.encode_boosts(
            history_boosts[:, :, -1] * history_mask
        )
        history_field_encoding = self.encode_field(
            history_field[:, :, -1] * history_mask
        )
        history_context_encoding = torch.cat(
            (
                history_side_conditions_encoding.flatten(3),
                history_volatile_status_encoding.flatten(3),
                history_boosts_encoding.flatten(3),
                history_field_encoding,
            ),
            dim=-1,
        )
        history_context_embedding = self.context_linear(history_context_encoding)

        active_weight = active_token.unsqueeze(-2).float()
        active_embeddings = (active_weight @ entity_embeddings).squeeze(-2)

        entities_embedding = self.entities_to_vector(active_embeddings.flatten(-2))

        side_conditions_encoding = self.encode_side_conditions(
            side_conditions[:, :, -1]
        ).flatten(2)
        volatile_status_encoding = self.encode_volatile_status(
            volatile_status[:, :, -1]
        ).flatten(2)
        boosts_encoding = self.encode_boosts(boosts[:, :, -1]).flatten(2)
        field_encoding = self.encode_field(field[:, :, -1])

        context_encoding = torch.cat(
            (
                side_conditions_encoding,
                volatile_status_encoding,
                boosts_encoding,
                field_encoding,
            ),
            dim=-1,
        )
        context_embedding = self.context_linear(context_encoding)
        # context_embedding = self.context_resnet(context_embedding)
        state_embedding = self.torso_merge(
            {"entities": entities_embedding, "context": context_embedding}
        )
        # state_embedding = self.torso_resnet(state_embedding)

        active_movesets = (
            (active_weight @ teams[:, :, -1].float()).long()[..., -8:-4].squeeze(-2)
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

        actions_onehot = self.moves_onehot(active_movesets)
        user_embeddings = torch.cat(
            (
                active_embeddings.unsqueeze(-2).expand(-1, -1, -1, 4, -1),
                entity_embeddings,
            ),
            dim=-2,
        )
        action_query = state_embedding[..., None, None, :]
        # action_query = self.action_query_resnet(action_query)
        action_embeddings = self.action_merge(
            {"action": actions_onehot, "user": user_embeddings}
        )
        # action_embeddings = self.action_keys_resnet(action_embeddings)

        # logits = self.action_logits(action_embeddings).flatten(2)
        logits = self.action_logits(action_query, action_embeddings)
        my_logits = logits[:, :, 0].flatten(2)

        policy = _legal_policy(my_logits, legal)
        log_policy = _legal_log_policy(my_logits, legal)

        value_hidden = state_embedding
        # value_hidden = self.value_resnet(state_embedding)
        value = self.value_mlp(value_hidden)

        return ModelOutput(
            policy=policy,
            log_policy=log_policy,
            logits=my_logits,
            value=value,
        )
