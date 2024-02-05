import math

import numpy as np

from pokesim.data import SPECIES_STOI, load_gendata
from pokesim.embeddings.encoding_funcs import (
    concat_encodings,
    multihot_encode,
    onehot_encode,
    z_score_scale,
)
from pokesim.embeddings.helpers import to_id
from pokesim.embeddings.moves import get_df


ONEHOT_FEATURES = [
    "name",
    "nfe",
]

MULTIHOT_FEATURES = [
    "abilities",
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

SPECIES_PROTOCOLS = [
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


def get_typechart_df(gen: int):
    data = load_gendata(gen, "typechart")
    return get_df(data)


def construct_species_encoding(gen: int):
    typechart_df = get_typechart_df(gen)
    species_df = get_species_df(gen)

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
    placeholder[row_index] = concat_df.mean(0)

    return placeholder.astype(np.float32)


if __name__ == "__main__":
    print(construct_species_encoding(3).shape)
