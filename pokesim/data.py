import os
import yaml
import json
import numpy as np

from typing import Any, Dict


with open(os.path.abspath("./config.yml"), "r") as f:
    CONFIG: Dict[str, Any] = yaml.safe_load(f)

SOCKET_PATH = CONFIG["socket_path"]
ENCODING = CONFIG["encoding"]
NUM_WORKERS = CONFIG["num_workers"]
DEFAULT_WORKER_INDEX = CONFIG.get("default_worker_index")
RANDOM_WORKER_INDEX = CONFIG.get("random_worker_index")
PREV_WORKER_INDEX = CONFIG.get("prev_worker_index")
EVAL_WORKER_INDEX = min(DEFAULT_WORKER_INDEX, RANDOM_WORKER_INDEX, PREV_WORKER_INDEX)

EVAL_MAPPING = {
    DEFAULT_WORKER_INDEX: "default",
    RANDOM_WORKER_INDEX: "random",
    PREV_WORKER_INDEX: "prev",
}

TURN_OFFSET = 0
TURN_SIZE = 1

TEAM_OFFSET = TURN_OFFSET + TURN_SIZE
TEAM_SIZE = 3 * 6 * 22

SIDE_CONDITION_OFFSET = TEAM_OFFSET + TEAM_SIZE
SIDE_CONDITION_SIZE = 2 * 15

VOLATILE_STATUS_OFFSET = SIDE_CONDITION_OFFSET + SIDE_CONDITION_SIZE
VOLATILE_STATUS_SIZE = 2 * 20

BOOSTS_OFFSET = VOLATILE_STATUS_OFFSET + VOLATILE_STATUS_SIZE
BOOSTS_SIZE = 2 * 7

FIELD_OFFSET = BOOSTS_OFFSET + BOOSTS_SIZE
FIELD_SIZE = 9 + 6

HISTORY_OFFSET = FIELD_OFFSET + FIELD_SIZE


with open(os.path.abspath("./src/data.json"), "r") as f:
    DATA = json.load(f)

with open(os.path.abspath("./src/tokens.json"), "r") as f:
    TOKENS = json.load(f)


STATUS_MAPPING = {
    "slp": 0,
    "psn": 1,
    "brn": 2,
    "frz": 3,
    "par": 4,
    "tox": 5,
}

BOOSTS_MAPPING = {
    "atk": 0,
    "def": 1,
    "spa": 2,
    "spd": 3,
    "spe": 4,
    "accuracy": 5,
    "evasion": 6,
}


NUM_SPECIES = len(TOKENS["species"])
NUM_ABILITIES = len(TOKENS["abilities"])
NUM_ITEMS = len(TOKENS["items"])
NUM_MOVES = len(TOKENS["moves"])
NUM_TYPES = len(TOKENS["types"])

NUM_TERRAIN = len(DATA["terrain"])
NUM_VOLATILE_STATUS = len(DATA["volatileStatus"])
NUM_WEATHER = len(DATA["weathers"])
NUM_SIDE_CONDITIONS = len(DATA["sideConditions"])
NUM_PSEUDOWEATHER = len(DATA["pseudoWeather"])

MAX_HP = 1024
NUM_HP_BUCKETS = int(MAX_HP**0.5 + 1)

NUM_STATUS = len(STATUS_MAPPING)
NUM_BOOSTS = len(BOOSTS_MAPPING)


def get_positional_encoding_matrix(
    d_model: int = 64, max_len: int = 1200
) -> np.ndarray:
    position = np.arange(max_len)[..., None]
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe.astype(np.float32)


TURN_ENC_SIZE = 64
TURN_MAX = 100

NUM_PLAYERS = 2
NUM_HISTORY = 8

MODEL_INPUT_KEYS = {
    "turn",
    "active_moveset",
    "teams",
    "side_conditions",
    "volatile_status",
    "boosts",
    "field",
    "legal",
    "history",
    "history_mask",
}
