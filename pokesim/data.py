import os
import yaml
import json
import numpy as np

from typing import Any, Dict

STATE_SIZE = 9395


with open(os.path.abspath("./config.yml"), "r") as f:
    CONFIG: Dict[str, Any] = yaml.safe_load(f)

SOCKET_PATH = CONFIG["socket_path"]
ENCODING = CONFIG["encoding"]
NUM_WORKERS = CONFIG["num_workers"]

DEFAULT_WORKER_INDEX = CONFIG.get("default_worker_index")
if DEFAULT_WORKER_INDEX < 0:
    DEFAULT_WORKER_INDEX = NUM_WORKERS + DEFAULT_WORKER_INDEX

RANDOM_WORKER_INDEX = CONFIG.get("random_worker_index")
if RANDOM_WORKER_INDEX < 0:
    RANDOM_WORKER_INDEX = NUM_WORKERS + RANDOM_WORKER_INDEX

MAXDMG_WORKER_INDEX = CONFIG.get("maxdmg_worker_index")
if MAXDMG_WORKER_INDEX < 0:
    MAXDMG_WORKER_INDEX = NUM_WORKERS + MAXDMG_WORKER_INDEX

PREV_WORKER_INDEX = CONFIG.get("prev_worker_index")
if PREV_WORKER_INDEX < 0:
    PREV_WORKER_INDEX = NUM_WORKERS + PREV_WORKER_INDEX

HEURISTIC_WORKER_INDEX = CONFIG.get("heuristic_worker_index")
if HEURISTIC_WORKER_INDEX < 0:
    HEURISTIC_WORKER_INDEX = NUM_WORKERS + HEURISTIC_WORKER_INDEX


EVAL_WORKER_INDEX = min(
    DEFAULT_WORKER_INDEX,
    RANDOM_WORKER_INDEX,
    PREV_WORKER_INDEX,
    HEURISTIC_WORKER_INDEX,
    MAXDMG_WORKER_INDEX,
)

ACTION_SPACE = list(range(10))

EVAL_MAPPING = {
    DEFAULT_WORKER_INDEX: "default",
    RANDOM_WORKER_INDEX: "random",
    PREV_WORKER_INDEX: "prev",
    HEURISTIC_WORKER_INDEX: "heuristic",
    MAXDMG_WORKER_INDEX: "maxdmg",
}

TURN_OFFSET = 0
TURN_SIZE = 1

HEURISTIC_OFFSET = TURN_OFFSET + TURN_SIZE
HEURISTIC_SIZE = 1

TEAM_OFFSET = HEURISTIC_OFFSET + HEURISTIC_SIZE
TEAM_SIZE = 3 * 6 * 48

SIDE_CONDITION_OFFSET = TEAM_OFFSET + TEAM_SIZE
SIDE_CONDITION_SIZE = 2 * 16

VOLATILE_STATUS_OFFSET = SIDE_CONDITION_OFFSET + SIDE_CONDITION_SIZE
VOLATILE_STATUS_SIZE = 2 * 108

BOOSTS_OFFSET = VOLATILE_STATUS_OFFSET + VOLATILE_STATUS_SIZE
BOOSTS_SIZE = 2 * 7

PSEUDOWEATHER_OFFSET = BOOSTS_OFFSET + BOOSTS_SIZE
PSEUDOWEATHER_SIZE = 27

WEATHER_OFFSET = PSEUDOWEATHER_OFFSET + PSEUDOWEATHER_SIZE
WEATHER_SIZE = 3

TERRAIN_OFFSET = WEATHER_OFFSET + WEATHER_SIZE
TERRAIN_SIZE = 3

HISTORY_OFFSET = TERRAIN_OFFSET + TERRAIN_SIZE

HISTORY_SIDE_CONDITION_OFFSET = 0
HISTORY_VOLATILE_STATUS_OFFSET = HISTORY_SIDE_CONDITION_OFFSET + SIDE_CONDITION_SIZE
HISTORY_BOOSTS_OFFSET = HISTORY_VOLATILE_STATUS_OFFSET + VOLATILE_STATUS_SIZE
HISTORY_PSEUDOWEATHER_OFFSET = HISTORY_BOOSTS_OFFSET + BOOSTS_SIZE
HISTORY_WEATHER_OFFSET = HISTORY_PSEUDOWEATHER_OFFSET + PSEUDOWEATHER_SIZE
HISTORY_TERRAIN_OFFSET = HISTORY_WEATHER_OFFSET + WEATHER_SIZE


with open(os.path.abspath("./src/data/data.json"), "r") as f:
    DATA = json.load(f)


def load_gendata(gen: int, datum: str):
    with open(os.path.abspath(f"./src/data/gen{gen}/{datum}.json"), "r") as f:
        return json.load(f)


VERBOSE = False

SPECIES_STOI = DATA["species"]
MOVES_STOI = DATA["moves"]

NUM_SPECIES = len(SPECIES_STOI)
NUM_ABILITIES = len(DATA["abilities"])
NUM_ITEMS = len(DATA["items"])
NUM_MOVES = len(MOVES_STOI)

# NUM_TYPES = len(DATA["types"])

NUM_TERRAIN = len(DATA["terrain"])
NUM_VOLATILE_STATUS = len(DATA["volatileStatus"])
NUM_WEATHER = len(DATA["weathers"])
NUM_SIDE_CONDITIONS = len(DATA["sideConditions"])
NUM_PSEUDOWEATHER = len(DATA["pseudoWeather"])


MAX_HP = 1024
NUM_HP_BUCKETS = int(MAX_HP**0.5 + 1)

NUM_STATUS = len(DATA["statuses"])
NUM_BOOSTS = len(DATA["boosts"])

if VERBOSE:
    print(f"NUM_SPECIES: {NUM_SPECIES}")
    print(f"NUM_ABILITIES: {NUM_ABILITIES}")
    print(f"NUM_ITEMS: {NUM_ITEMS}")
    print(f"NUM_MOVES: {NUM_MOVES}")
    print(f"NUM_TERRAIN: {NUM_TERRAIN}")
    print(f"NUM_VOLATILE_STATUS: {NUM_VOLATILE_STATUS}")
    print(f"NUM_WEATHER: {NUM_WEATHER}")
    print(f"NUM_SIDE_CONDITIONS: {NUM_SIDE_CONDITIONS}")
    print(f"NUM_PSEUDOWEATHER: {NUM_PSEUDOWEATHER}")
    print(f"NUM_STATUS: {NUM_STATUS}")
    print(f"NUM_BOOSTS: {NUM_BOOSTS}")


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
TURN_MAX = 127

NUM_PLAYERS = 2
NUM_HISTORY = 8

MODEL_INPUT_KEYS = {
    "turn",
    "teams",
    "side_conditions",
    "volatile_status",
    "boosts",
    "pseudoweather",
    "weather",
    "terrain",
    "history_side_conditions",
    "history_volatile_status",
    "history_boosts",
    "history_pseudoweather",
    "history_weather",
    "history_terrain",
    "history_entities",
    "history_stats",
    "legal",
}
