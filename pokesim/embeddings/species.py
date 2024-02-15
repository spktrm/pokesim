from functools import partial
import math
import numpy as np
import pandas as pd

from typing import List

from pokesim.data import SPECIES_STOI, load_gendata
from pokesim.embeddings.encoding_funcs import (
    concat_encodings,
    multihot_encode,
    onehot_encode,
    sqrt_onehot_encode,
    z_score_scale,
)
from pokesim.embeddings.helpers import Protocol, to_id
from pokesim.embeddings.moves import get_df


ONEHOT_FEATURES = [
    "name",
    # "nfe",
]

MULTIHOT_FEATURES = [
    # "abilities",
    # "types",
]


STAT_FEATURES = [
    # "baseStats.hp",
    # "baseStats.atk",
    # "baseStats.def",
    # "baseStats.spa",
    # "baseStats.spd",
    # "baseStats.spe",
]


def encode_continuous_values(series: pd.Series, n_bins):
    values = series.values
    range = series.max() - series.min()
    arr = np.arange(n_bins)[None] < np.floor(n_bins * values / range)[:, None]
    arr = arr.astype(float)
    extra = (values % (range / n_bins)) / (range / n_bins)
    extra_mask = (
        np.arange(n_bins)[None] <= np.floor(n_bins * values / range)[:, None]
    ) - arr
    arr = arr + extra_mask * extra[:, None]
    return pd.DataFrame(data=arr)


SPECIES_PROTOCOLS: List[Protocol] = [
    *[
        {
            "feature": stat_feature,
            "func": partial(encode_continuous_values, n_bins=16),
        }
        for stat_feature in STAT_FEATURES
    ],
    # *[
    #     {"feature": stat_feature, "func": z_score_scale}
    #     for stat_feature in STAT_FEATURES
    # ],
    {
        "feature": "weightkg",
        "func": lambda series: z_score_scale(series.map(math.log)),
    },
    *[
        {"feature": stat_feature, "func": onehot_encode}
        for stat_feature in ONEHOT_FEATURES
    ],
    *[
        {"feature": stat_feature, "func": multihot_encode}
        for stat_feature in MULTIHOT_FEATURES
    ],
]


def get_species_df(gen: int):
    data = load_gendata(gen, "species")

    df = get_df(data)
    df = df.sort_values("num")

    ability_columns = [
        column for column in df.columns if column.startswith("abilities")
    ]
    df["abilities"] = df.fillna("").apply(
        lambda row: [
            value for value in [row[column] for column in ability_columns] if value
        ],
        axis=1,
    )
    return df


def get_learnset_df(gen: int, species_df: pd.DataFrame):
    data = load_gendata(gen, "learnsets")
    for pokemon in data:
        moves_to_pop = []

        for move, learnset in pokemon.get("learnset", {}).items():
            if not any([option.startswith(f"{gen}") for option in learnset]):
                moves_to_pop.append(move)
            else:
                pokemon["learnset"][move] = True

        for move in moves_to_pop:
            pokemon["learnset"].pop(move)

    return get_df(data)


def get_typechart_df(gen: int):
    data = load_gendata(gen, "typechart")
    return get_df(data)


def construct_species_encoding(gen: int):
    # typechart_df = get_typechart_df(gen)
    species_df = get_species_df(gen)
    # learnset_df = get_learnset_df(gen, species_df)

    feature_vector_dfs = []

    for protoc in SPECIES_PROTOCOLS:
        func = protoc["func"]
        feature = protoc["feature"]
        feature_vector_dfs.append(func(species_df[feature]))

    concat_df = concat_encodings(feature_vector_dfs)
    concat_df.index = species_df["name"].map(to_id)

    placeholder = np.zeros((len(SPECIES_STOI), concat_df.shape[-1]))

    for name, row in concat_df.iterrows():
        row_index = SPECIES_STOI[name]
        placeholder[row_index] = row

    row_index = SPECIES_STOI["<UNK>"]
    placeholder[row_index] = 1

    return placeholder.astype(np.float32)


if __name__ == "__main__":
    print(construct_species_encoding(3).shape)
