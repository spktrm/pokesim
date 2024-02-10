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
    # "name",
    "nfe",
]

MULTIHOT_FEATURES = [
    # "abilities",
    "types",
]


STAT_FEATURES = [
    "baseStats.hp",
    "baseStats.atk",
    "baseStats.def",
    "baseStats.spa",
    "baseStats.spd",
    "baseStats.spe",
]


def get_type_data(series: pd.Series):
    typechart_df = get_typechart_df(3)

    type_encoding = (
        multihot_encode(series).values[:, None, :] * typechart_df.values[None]
    )
    type_encoding = type_encoding.swapaxes(1, 2)
    type_encoding[type_encoding.sum(-1) == 0] = 1
    type_encoding = type_encoding.prod(1)
    return


SPECIES_PROTOCOLS: List[Protocol] = [
    # *[
    #     {"feature": stat_feature, "func": sqrt_onehot_encode}
    #     for stat_feature in STAT_FEATURES
    # ],
    *[
        {"feature": stat_feature, "func": z_score_scale}
        for stat_feature in STAT_FEATURES
    ],
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
    df = get_df(data)

    df = df[[c for c in df.columns if c.startswith("damageTaken.")]]
    if gen < 6:
        df = df.drop("damageTaken.Fairy", axis=1)
    if gen < 2:
        df = df.drop("damageTaken.Steel", axis=1)
        df = df.drop("damageTaken.Dark", axis=1)

    cols_to_drop = []
    for col in df.columns:
        if len(df[col].unique()) == 1:
            cols_to_drop.append(col)

    df = df[[c for c in df.columns if c not in cols_to_drop]]

    df = df.replace({0: 1, 1: 2, 2: 0.5, 3: 0})
    return df


def construct_species_encoding(gen: int):
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

    valid_indices = []
    for name, row in concat_df.iterrows():
        row_index = SPECIES_STOI[name]
        placeholder[row_index] = row
        valid_indices.append(row_index)

    row_index = SPECIES_STOI["<UNK>"]
    placeholder[row_index] = placeholder[np.array(valid_indices)].mean(0)

    return placeholder.astype(np.float32)


if __name__ == "__main__":
    print(construct_species_encoding(3).shape)
