from typing import Callable, List, TypedDict
import numpy as np

from pokesim.data import ITEMS_STOI, load_gendata
from pokesim.embeddings.encoding_funcs import (
    concat_encodings,
    multihot_encode,
    onehot_encode,
    z_score_scale,
)
from pokesim.embeddings.helpers import to_id, Protocol
from pokesim.embeddings.moves import get_df


ONEHOT_FEATURES = ["id"]

MULTIHOT_FEATURES = []


ITEMS_PROTOCOLS: List[Protocol] = [
    *[
        {"feature": stat_feature, "func": onehot_encode}
        for stat_feature in ONEHOT_FEATURES
    ],
    {"feature_fn": lambda x: x.startswith("fling."), "func": onehot_encode},
    {"feature_fn": lambda x: x.startswith("on"), "func": onehot_encode},
    {"feature_fn": lambda x: x.startswith("is"), "func": onehot_encode},
    {"feature_fn": lambda x: x.startswith("naturalGift."), "func": onehot_encode},
    *[
        {"feature": stat_feature, "func": multihot_encode}
        for stat_feature in MULTIHOT_FEATURES
    ],
]


def get_items_df(gen: int):
    data = load_gendata(gen, "items")

    df = get_df(data)
    df = df.sort_values("num")

    return df


def construct_items_encoding(gen: int):
    df = get_items_df(gen)

    feature_vector_dfs = []

    for protoc in ITEMS_PROTOCOLS:
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

    placeholder = np.zeros((len(ITEMS_STOI), concat_df.shape[-1]))

    for name, row in concat_df.iterrows():
        row_index = ITEMS_STOI[name]
        placeholder[row_index] = row

    row_index = ITEMS_STOI["<UNK>"]
    placeholder[row_index] = concat_df.mean(0).values

    return placeholder.astype(np.float32)


if __name__ == "__main__":
    print(construct_items_encoding(9).shape)
