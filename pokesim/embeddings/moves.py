import numpy as np

from pokesim.data import MOVES_STOI, SPECIES_STOI, load_gendata
from pokesim.embeddings.encoding_funcs import (
    concat_encodings,
    multihot_encode,
    onehot_encode,
    sqrt_onehot_encode,
    z_score_scale,
)
from pokesim.embeddings.helpers import get_df, to_id


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

MOVES_PROTOCOLS = [
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
    {"feature": "basePower", "func": sqrt_onehot_encode},
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


def construct_move_encoding(gen: int):
    moves_df = get_moves_df(gen)

    feature_vector_dfs = []

    for protoc in MOVES_PROTOCOLS:
        func = protoc["func"]
        feature = protoc.get("feature")
        feature_fn = protoc.get("feature_fn")

        if feature_fn is None:
            feature_df = func(moves_df[feature])
            if not feature_df.empty:
                feature_vector_dfs.append(feature_df)
        else:
            for column in moves_df.columns:
                if feature_fn(column):
                    feature_df = func(moves_df[column])
                    if not feature_df.empty:
                        feature_vector_dfs.append(feature_df)

    concat_df = concat_encodings(feature_vector_dfs)
    concat_df.index = moves_df["name"].map(to_id)

    placeholder = np.zeros((len(MOVES_STOI), concat_df.shape[-1]+1))

    for name, row in concat_df.iterrows():
        row_index = MOVES_STOI[name]
        placeholder[row_index, 1:] = row

    row_index = MOVES_STOI["<UNK>"]
    placeholder[row_index, 1:] = concat_df.mean(0)

    row_index = MOVES_STOI["<SWITCH>"]
    placeholder[row_index, 0] = 1

    return placeholder.astype(np.float32)


if __name__ == "__main__":
    print(construct_move_encoding(3).shape)
