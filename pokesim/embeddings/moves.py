import numpy as np
import pandas as pd

from typing import List

from pokesim.data import MOVES_STOI, load_gendata
from pokesim.embeddings.encoding_funcs import (
    concat_encodings,
    multihot_encode,
    onehot_encode,
    sqrt_onehot_encode,
    z_score_scale,
)
from pokesim.embeddings.helpers import Protocol, get_df, to_id


ONEHOT_FEATURES = [
    "category",
    "priority",
    "type",
    "name",
    "target",
    "volatileStatus",
    "status",
    "breaksProtect",
    "weather",
    "stallingMove",
    "sleepUsable",
    "selfdestruct",
    "struggleRecoil",
    "smartTarget",
    "slotCondition",
    "stealsBoosts",
    "terrain",
    "forceSwitch",
    "hasCrashDamage",
    "hasSheerForce",
    "mindBlownRecoil",
    "onDamagePriority",
    "onTry",
    "recoil",
    "heal",
    "ohko",
]

MULTIHOT_FEATURES = []

MOVES_PROTOCOLS: List[Protocol] = [
    *[
        {"feature": stat_feature, "func": onehot_encode}
        for stat_feature in ONEHOT_FEATURES
    ],
    *[
        {"feature": stat_feature, "func": multihot_encode}
        for stat_feature in MULTIHOT_FEATURES
    ],
    {"feature_fn": lambda x: x.startswith("flags."), "func": lambda x: x.fillna(0)},
    {"feature_fn": lambda x: x.startswith("condition."), "func": onehot_encode},
    {"feature_fn": lambda x: x.startswith("boosts."), "func": onehot_encode},
    {"feature_fn": lambda x: x.startswith("secondary."), "func": onehot_encode},
    {"feature_fn": lambda x: x.startswith("self."), "func": onehot_encode},
    {"feature_fn": lambda x: x.startswith("selfBoost."), "func": onehot_encode},
    {"feature_fn": lambda x: x.startswith("ignore"), "func": onehot_encode},
    {"feature": "basePower", "func": z_score_scale},
    # {"feature": "basePower", "func": sqrt_onehot_encode},
    {
        "feature": "accuracy",
        "func": lambda x: (x.map(lambda v: 100 if isinstance(v, bool) else v) / 100),
    },
    {
        "feature": "accuracy",
        "func": lambda x: x.map(lambda v: 1 if isinstance(v, bool) else 0),
    },
]


def get_moves_df(gen: int):
    data = load_gendata(gen, "moves")

    df = get_df(data)
    df = df.sort_values("num")

    return df


def get_typechart_df(gen: int):
    data = load_gendata(gen, "typechart")
    df = get_df(data)

    effectiveness = concat_encodings(
        [
            onehot_encode(df[feature])
            for feature in df.columns
            if feature.startswith("damageTaken.")
        ]
    )
    effectiveness.index = df.index
    return effectiveness


def construct_moves_encoding(gen: int):
    typechart_df = get_typechart_df(gen)
    moves_df = get_moves_df(gen)

    typechart_df = pd.DataFrame(
        data=np.stack(
            moves_df["type"]
            .map(to_id)
            .map(
                lambda x: (
                    np.zeros_like(typechart_df.loc["normal"].values)
                    if x == ""
                    else typechart_df.loc[x].values
                )
            )
            .values
        )
    )

    feature_vector_dfs = []  # [typechart_df]

    for protoc in MOVES_PROTOCOLS:
        func = protoc["func"]
        feature = protoc.get("feature")
        feature_fn = protoc.get("feature_fn")

        if feature_fn is None:

            if feature not in moves_df.columns:
                print(f"{feature} not in df")
                continue
            series = moves_df[feature]
            feature_df = func(series)
            if not feature_df.empty:
                feature_vector_dfs.append(feature_df)
        else:
            for feature in moves_df.columns:
                if feature not in moves_df.columns:
                    print(f"{feature} not in df")
                    continue
                if feature_fn(feature):
                    series = moves_df[feature]
                    feature_df = func(series)
                    if not feature_df.empty:
                        feature_vector_dfs.append(feature_df)

    concat_df = concat_encodings(feature_vector_dfs)
    concat_df.index = moves_df["name"].map(to_id)

    placeholder = np.zeros((len(MOVES_STOI), concat_df.shape[-1] + 1))

    for name, row in concat_df.iterrows():
        row_index = MOVES_STOI[name]
        placeholder[row_index, 1:] = row

    placeholder[row_index, 1:] = placeholder[MOVES_STOI["return102"], 1:]

    row_index = MOVES_STOI["<UNK>"]
    placeholder[row_index, 1:] = concat_df.mean(0).values

    row_index = MOVES_STOI["<SWITCH>"]
    placeholder[row_index, 0] = 1

    return placeholder.astype(np.float32)


if __name__ == "__main__":
    print(construct_moves_encoding(3).shape)
