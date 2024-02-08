import numpy as np

from typing import List

from pokesim.data import ABILITIES_STOI, load_gendata
from pokesim.embeddings.encoding_funcs import (
    concat_encodings,
    onehot_encode,
)
from pokesim.embeddings.helpers import to_id, Protocol
from pokesim.embeddings.moves import get_df


ONEHOT_FEATURES = ["id", "suppressWeather"]

ABILITIES_PROTOCOLS: List[Protocol] = [
    *[
        {"feature": stat_feature, "func": onehot_encode}
        for stat_feature in ONEHOT_FEATURES
    ],
    {"feature_fn": lambda x: x.startswith("condition."), "func": onehot_encode},
    {"feature_fn": lambda x: x.startswith("on"), "func": onehot_encode},
    {"feature_fn": lambda x: x.startswith("is"), "func": onehot_encode},
]


def get_abilities_df(gen: int):
    data = load_gendata(gen, "abilities")

    df = get_df(data)
    df = df.sort_values("num")

    return df


def construct_abilities_encoding(gen: int):
    df = get_abilities_df(gen)

    feature_vector_dfs = []

    for protoc in ABILITIES_PROTOCOLS:
        func = protoc["func"]
        feature = protoc.get("feature")
        feature_fn = protoc.get("feature_fn")

        if feature_fn is None:

            if feature not in df.columns:
                print(f"{feature} not in df")
                continue
            series = df[feature]
            feature_df = func(series)
            if not feature_df.empty:
                feature_vector_dfs.append(feature_df)
        else:
            for feature in df.columns:
                if feature not in df.columns:
                    print(f"{feature} not in df")
                    continue
                if feature_fn(feature):
                    series = df[feature]
                    feature_df = func(series)
                    if not feature_df.empty:
                        feature_vector_dfs.append(feature_df)

    concat_df = concat_encodings(feature_vector_dfs)
    concat_df.index = df["name"].map(to_id)

    placeholder = np.zeros((len(ABILITIES_STOI), concat_df.shape[-1]))

    for name, row in concat_df.iterrows():
        row_index = ABILITIES_STOI[name]
        placeholder[row_index] = row

    row_index = ABILITIES_STOI["<UNK>"]
    placeholder[row_index] = concat_df.mean(0).values

    return placeholder.astype(np.float32)


if __name__ == "__main__":
    print(construct_abilities_encoding(9).shape)
