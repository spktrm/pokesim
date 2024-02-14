import numpy as np
import torch
import torch.nn as nn

from pokesim.data import (
    MOVES_STOI,
    NUM_ABILITIES,
    NUM_BOOSTS,
    NUM_PSEUDOWEATHER,
    NUM_SIDE_CONDITIONS,
    NUM_STATUS,
    NUM_TERRAIN,
    NUM_VOLATILE_STATUS,
    NUM_WEATHER,
    SIDE_CONDITIONS_STOI,
    SPECIES_STOI,
)

from typing import Tuple

from pokesim.structs import ModelOutput
from pokesim.utils import _legal_log_policy, _legal_policy, _print_params

from pokesim.nn.helpers import _layer_init
from pokesim.nn.modules import (
    GatingType,
    MultiHeadAttention,
    PointerLogits,
    Resnet,
    ToVector,
    Transformer,
    TransformerEncoder,
    VectorMerge,
    MLP,
)

# from pokemb.mod import ComponentEmbedding


PADDING_TOKEN = SPECIES_STOI["<PAD>"]
UNKNOWN_TOKEN = SPECIES_STOI["<UNK>"]
SWITCH_TOKEN = MOVES_STOI["<SWITCH>"]


def get_multihot_scalar_encoding(num_tokens, n_bins):
    range = num_tokens
    values = np.arange(num_tokens)
    arr = np.arange(n_bins)[None] < np.floor(n_bins * values / range)[:, None]
    arr = arr.astype(float)
    extra = (values % (range / n_bins)) / (range / n_bins)
    extra = 2 * extra - 1
    extra_mask = (
        np.arange(n_bins)[None] <= np.floor(n_bins * values / range)[:, None]
    ) - arr
    arr = arr + extra_mask * extra[:, None]
    arr = np.where(arr <= 0, -1, arr)
    return nn.Embedding.from_pretrained(torch.from_numpy(arr).to(torch.float))


def get_onehot_encoding(num_embeddings: int):
    return nn.Embedding.from_pretrained(torch.eye(num_embeddings))


class CustomEmbedding(nn.Module):
    def __init__(self, gen: int, category: str, entity_size: int):
        super().__init__()

        npy_arr = np.load(f"src/data/gen{gen}/{category}.npy")
        self._embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(npy_arr).to(torch.float32)
        )
        self._linear = _layer_init(nn.Linear(npy_arr.shape[-1], entity_size))

    def forward(self, x: torch.Tensor):
        x = self._embedding(x)
        return self._linear(x)


class Encoder(nn.Module):
    def __init__(
        self,
        entity_size: int,
        stream_size: int,
        use_layer_norm: bool,
        affine_layer_norm: bool,
        gen: int = 3,
    ):
        super().__init__()

        self.species_onehot = CustomEmbedding(gen, "species", entity_size)
        self.item_onehot = CustomEmbedding(gen, "items", entity_size)
        self.ability_onehot = CustomEmbedding(gen, "abilities", entity_size)
        self.moves_onehot = CustomEmbedding(gen, "moves", entity_size)

        self.hp_onehot = get_multihot_scalar_encoding(1024, 64)
        self.status_onehot = get_onehot_encoding(NUM_STATUS)
        self.onehot2 = get_onehot_encoding(2)
        self.onehot4 = get_onehot_encoding(4)
        self.toxic_turns_onehot = get_onehot_encoding(6)

        rest_size = (
            entity_size
            + self.hp_onehot.weight.shape[-1]
            + self.status_onehot.weight.shape[-1]
            + self.onehot2.weight.shape[-1]
            + self.onehot2.weight.shape[-1]
            + self.onehot2.weight.shape[-1]
            + self.onehot2.weight.shape[-1]
            + self.onehot4.weight.shape[-1]
            + self.toxic_turns_onehot.weight.shape[-1]
        )
        self.rest_lin = _layer_init(nn.Linear(rest_size, entity_size))
        self.entity_merge = VectorMerge(
            input_sizes={
                "species": entity_size,
                "ability": entity_size,
                "item": entity_size,
                "moveset": entity_size,
                "rest": entity_size,
            },
            output_size=entity_size,
            gating_type=GatingType.NONE,
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )

        self.boosts_onehot = get_onehot_encoding(13)

        self.tspikes_onehot = get_onehot_encoding(3)
        # self.volatile_status_onehot = get_onehot_embedding(NUM_VOLATILE_STATUS)

        # self.pseudoweathers_onehot = get_onehot_embedding(NUM_PSEUDOWEATHER)
        self.weathers_onehot = get_onehot_encoding(NUM_WEATHER)
        self.terrain_onehot = get_onehot_encoding(NUM_TERRAIN)
        self.min_max_duration_onehot = get_onehot_encoding(10)

        weather_size = (
            self.weathers_onehot.weight.shape[-1]
            + 2 * self.min_max_duration_onehot.weight.shape[-1]
        )
        terrain_size = (
            self.terrain_onehot.weight.shape[-1]
            + 2 * self.min_max_duration_onehot.weight.shape[-1]
        )
        field_size = NUM_PSEUDOWEATHER + weather_size + terrain_size

        context_size = (
            2
            * (
                NUM_BOOSTS * self.boosts_onehot.weight.shape[-1]
                + NUM_VOLATILE_STATUS
                + NUM_SIDE_CONDITIONS
                + self.onehot4.weight.shape[-1]
                + self.tspikes_onehot.weight.shape[-1]
            )
            + field_size
        )

        self.context_lin = nn.Sequential(
            _layer_init(nn.Linear(context_size, 2 * entity_size)),
        )
        self.context_mlp = MLP([2 * entity_size, stream_size])

        self.entities_cls = nn.Parameter(
            torch.randn(1, 1, 4, entity_size) / (entity_size**0.5)
        )
        self.entity_transformer = TransformerEncoder(
            units_stream_size=entity_size,
            transformer_num_layers=1,
            transformer_num_heads=2,
            transformer_key_size=entity_size // 2,
            transformer_value_size=entity_size // 2,
            resblocks_num_before=1,
            resblocks_num_after=1,
            resblocks_hidden_size=entity_size // 2,
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )

    def forward_entities(self, entities: torch.Tensor) -> torch.Tensor:
        species_token = entities[..., 0]
        item_token = entities[..., 1]
        ability_token = entities[..., 2]
        hp_token = entities[..., 3]
        active_token = entities[..., 4]
        fainted_token = entities[..., 5]
        status_token = entities[..., 6]
        last_move_token = entities[..., 7]
        public_token = entities[..., 8]
        side_token = entities[..., 9]
        sleep_turns_token = entities[..., 10].clamp(min=0, max=3)
        toxic_turns_token = entities[..., 11].clamp(min=0, max=5)
        move_pp_left = entities[..., 12:16]
        move_pp_max = entities[..., 16:20]
        move_tokens = entities[..., 20:]

        species_onehot = self.species_onehot(species_token)
        item_onehot = self.item_onehot(item_token)
        ability_onehot = self.ability_onehot(ability_token)
        hp_onehot = self.hp_onehot(hp_token)
        active_onehot = self.onehot2(active_token)
        fainted_onehot = self.onehot2(fainted_token)
        status_onehot = self.status_onehot(status_token)
        last_move_onehot = self.moves_onehot(last_move_token)
        side_onehot = self.onehot2(side_token)
        public_onehot = self.onehot2(public_token)
        sleep_turns_onehot = self.onehot4(sleep_turns_token)
        toxic_turns_onehot = self.toxic_turns_onehot(toxic_turns_token)
        moveset_onehot = self.moves_onehot(move_tokens).mean(-2)

        rest_onehot = torch.cat(
            (
                # species_onehot,
                # item_onehot,
                # ability_onehot,
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
                # moveset_onehot,
            ),
            dim=-1,
        )
        raw_embeddings = self.entity_merge(
            {
                "species": species_onehot,
                "ability": ability_onehot,
                "item": item_onehot,
                "moveset": moveset_onehot,
                "rest": self.rest_lin(rest_onehot),
            }
        )

        return raw_embeddings

    def encode_side_conditions(self, side_conditions: torch.Tensor) -> torch.Tensor:
        spikes_token = side_conditions[..., SIDE_CONDITIONS_STOI["spikes"]]
        tspikes_token = side_conditions[..., SIDE_CONDITIONS_STOI["toxicspikes"]]
        other = side_conditions > 0
        spikes = self.onehot4(spikes_token)
        tspikes = self.tspikes_onehot(tspikes_token)
        return torch.cat((other, spikes, tspikes), -1)

    def encode_volatile_status(self, volatile_status: torch.Tensor) -> torch.Tensor:
        encoding = (volatile_status > 0).to(torch.long)
        return encoding

    def encode_boosts(self, boosts: torch.Tensor) -> torch.Tensor:
        return self.boosts_onehot(boosts + 6)

    def encode_pseudoweather(self, pseudoweather: torch.Tensor) -> torch.Tensor:
        id = pseudoweather[..., 0]
        min_durr = pseudoweather[..., 1]
        max_durr = pseudoweather[..., 2]

        min_durr_onehot = self.min_max_duration_onehot(min_durr)
        max_durr_onehot = self.min_max_duration_onehot(max_durr)

        return id  # torch.cat((id, min_durr_onehot, max_durr_onehot), -1)

    def encode_weather(self, weather: torch.Tensor) -> torch.Tensor:
        id = weather[..., 0]
        min_durr = weather[..., 1]
        max_durr = weather[..., 2]

        id_onehot = self.weathers_onehot(id)
        min_durr_onehot = self.min_max_duration_onehot(min_durr)
        max_durr_onehot = self.min_max_duration_onehot(max_durr)

        return torch.cat((id_onehot, min_durr_onehot, max_durr_onehot), -1)

    def encode_terrain(self, terrain: torch.Tensor) -> torch.Tensor:
        id = terrain[..., 0]
        min_durr = terrain[..., 1]
        max_durr = terrain[..., 2]

        id_onehot = self.terrain_onehot(id)
        min_durr_onehot = self.min_max_duration_onehot(min_durr)
        max_durr_onehot = self.min_max_duration_onehot(max_durr)

        return torch.cat((id_onehot, min_durr_onehot, max_durr_onehot), -1)

    def embed_context(
        self,
        side_conditions: torch.Tensor,
        volatile_status: torch.Tensor,
        boosts: torch.Tensor,
        pseudoweather: torch.Tensor,
        weather: torch.Tensor,
        terrain: torch.Tensor,
    ) -> torch.Tensor:
        side_conditions_encoding = self.encode_side_conditions(side_conditions)
        volatile_status_encoding = self.encode_volatile_status(volatile_status)
        boosts_encoding = self.encode_boosts(boosts)
        pseudoweather_encoding = self.encode_pseudoweather(pseudoweather)
        weather_encoding = self.encode_weather(weather)
        terrain_encoding = self.encode_terrain(terrain)
        context_encoding = torch.cat(
            (
                side_conditions_encoding.flatten(-2),
                volatile_status_encoding.flatten(-2),
                boosts_encoding.flatten(-3),
                pseudoweather_encoding,
                weather_encoding,
                terrain_encoding,
            ),
            dim=-1,
        )
        return self.context_lin(context_encoding)

    def forward(
        self,
        teams: torch.Tensor,
        side_conditions: torch.Tensor,
        volatile_status: torch.Tensor,
        boosts: torch.Tensor,
        pseudoweather: torch.Tensor,
        weather: torch.Tensor,
        terrain: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        entity_embeddings = self.forward_entities(teams[:, :, -1])
        entity_embeddings_flat = entity_embeddings.flatten(2, 3)

        entity_embeddings_flat = torch.cat(
            (
                self.entities_cls.expand(*entity_embeddings_flat.shape[:2], -1, -1),
                entity_embeddings_flat,
            ),
            dim=-2,
        )
        entity_embeddings_flat = self.entity_transformer(
            entity_embeddings_flat,
            torch.ones_like(entity_embeddings_flat[..., 0], dtype=torch.bool),
        )
        entities_embedding = entity_embeddings_flat[..., :4, :].flatten(-2)
        entity_embeddings_flat = entity_embeddings_flat[..., 4:, :]

        active_token = teams[..., -1, :, :, 4]

        active_weight = active_token[..., 0, :].unsqueeze(-2).float()
        active_embedding = active_weight @ entity_embeddings_flat[..., :6, :]

        all_entity_embeddings = entity_embeddings
        my_entity_embeddings = entity_embeddings_flat[..., :6, :]

        context_encoding = self.embed_context(
            side_conditions[:, :, -1],
            volatile_status[:, :, -1],
            boosts[:, :, -1],
            pseudoweather[:, :, -1],
            weather[:, :, -1],
            terrain[:, :, -1],
        )
        context_embedding = self.context_mlp(context_encoding)

        return (
            active_weight,
            active_embedding,
            my_entity_embeddings,
            all_entity_embeddings,
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
        self.torso_resnet = Resnet(
            input_size=stream_size,
            num_resblocks=2,
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )

    def forward(
        self, entities_embedding: torch.Tensor, context_embedding: torch.Tensor
    ) -> torch.Tensor:
        state_embedding = self.torso_merge(
            {"entities": entities_embedding, "context": context_embedding}
        )
        state_embedding = self.torso_resnet(state_embedding)
        return state_embedding


class PolicyHead(nn.Module):
    def __init__(
        self,
        entity_size: int,
        stream_size: int,
        use_layer_norm: bool,
        affine_layer_norm: bool,
        gen: int = 3,
    ):
        super().__init__()

        switch_tokens = SWITCH_TOKEN * torch.ones(6, dtype=torch.long)
        self.register_buffer("switch_tokens", switch_tokens)

        self.moves_onehot = CustomEmbedding(gen, "moves", entity_size)
        self.legal_embedding = nn.Embedding(2, entity_size)

        self.action_merge = VectorMerge(
            {
                "action": entity_size,
                "user": entity_size,
                "legal": entity_size,
            },
            output_size=entity_size,
            gating_type=GatingType.NONE,
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )

        self.action_transformer = TransformerEncoder(
            units_stream_size=entity_size,
            transformer_num_layers=1,
            transformer_num_heads=2,
            transformer_key_size=entity_size // 2,
            transformer_value_size=entity_size // 2,
            resblocks_num_before=1,
            resblocks_num_after=1,
            resblocks_hidden_size=entity_size // 2,
            use_layer_norm=use_layer_norm,
            affine_layer_norm=affine_layer_norm,
        )

        self.action_logits = PointerLogits(
            stream_size,
            entity_size,
            num_layers_query=1,
            num_layers_keys=0,
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

    def embed_actions(self, active_movesets: torch.Tensor) -> torch.Tensor:
        moveset_pp_left = active_movesets[..., :4]
        moveset_pp_max = active_movesets[..., 4:8]
        moveset_tokens = active_movesets[..., 8:]

        expanded_switch_tokens = self.switch_tokens.view(1, 1, -1).expand(
            *active_movesets.shape[:2], 6
        )
        action_tokens = torch.cat((moveset_tokens, expanded_switch_tokens), dim=-1)

        return self.moves_onehot(action_tokens)

    def forward(
        self,
        active_weight: torch.Tensor,
        teams: torch.Tensor,
        legal: torch.Tensor,
        active_embedding: torch.Tensor,
        my_entity_embeddings: torch.Tensor,
        state_embedding: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        action_tokens = self.get_action_tokens(active_weight, teams)
        action_embeddings = self.embed_actions(action_tokens)

        user_embeddings = torch.cat(
            (active_embedding.expand(-1, -1, 4, -1), my_entity_embeddings), dim=-2
        )

        legal_embedding = self.legal_embedding(legal.to(torch.long))

        # action_query = self.action_query_resnet(action_query)
        action_embeddings = self.action_merge(
            {
                "action": action_embeddings,
                "user": user_embeddings,
                "legal": legal_embedding,
            }
        )

        # action_embeddings = self.action_keys_resnet(action_embeddings)
        action_embeddings = self.action_transformer(
            action_embeddings,
            torch.ones_like(action_embeddings[..., 0], dtype=torch.bool),
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
        entity_size: int,
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

        # self.score = MLP(
        #     [2 * entity_size, entity_size, 1],
        #     use_layer_norm=use_layer_norm,
        #     affine_layer_norm=affine_layer_norm,
        # )

    def forward(
        self,
        state_embedding: torch.Tensor,
        all_entity_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        value_hidden = state_embedding
        # value_hidden = self.value_resnet(state_embedding)

        # my_embeddings = (
        #     all_entity_embeddings[:, :, 0].unsqueeze(-2).expand(-1, -1, -1, 6, -1)
        # ).flatten(-3, -2)
        # opp_embeddings = (
        #     all_entity_embeddings[:, :, -1].unsqueeze(-3).expand(-1, -1, 6, -1, -1)
        # ).flatten(-3, -2)

        # matchup = torch.cat((my_embeddings, opp_embeddings), dim=-1)
        # scores = self.score(matchup)
        # embedding_score = scores.flatten(-2).mean(-1, keepdim=True)

        value = self.value_mlp(value_hidden)  # + embedding_score
        return value


class Model(nn.Module):
    def __init__(
        self,
        entity_size: int = 32,
        stream_size: int = 128,
        scale: int = 8,
        use_layer_norm: bool = True,
        affine_layer_norm: bool = True,
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
            entity_size=entity_size,
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
        pseudoweather: torch.Tensor,
        weather: torch.Tensor,
        terrain: torch.Tensor,
        legal: torch.Tensor,
    ):
        (
            active_weight,
            active_embedding,
            my_entity_embeddings,
            all_entity_embeddings,
            entities_embedding,
            context_embedding,
        ) = self.encoder.forward(
            teams=teams,
            side_conditions=side_conditions,
            volatile_status=volatile_status,
            boosts=boosts,
            pseudoweather=pseudoweather,
            weather=weather,
            terrain=terrain,
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
            my_entity_embeddings=my_entity_embeddings,
            state_embedding=state_embedding,
        )

        value = self.value_head.forward(
            state_embedding=state_embedding, all_entity_embeddings=all_entity_embeddings
        )

        return ModelOutput(
            policy=policy,
            log_policy=log_policy,
            logits=logits,
            value=value,
        )


if __name__ == "__main__":
    model = Model()
    _print_params(model)
