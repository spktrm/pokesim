import math
from typing import Callable, Sequence
import pandas as pd


def onehot_encode(series: pd.Series) -> pd.DataFrame:
    series = series.astype(str)
    unique_categories = [
        category for category in sorted(series.unique())  # if category not in ["nan"]
    ]
    records = series.map(
        lambda category: {
            f"{series.name}: {unique_category}": int(unique_category == category)
            for unique_category in unique_categories
        }
    ).tolist()
    dataframe = pd.DataFrame.from_records(records)
    return dataframe[list(sorted(dataframe.columns))].astype(float)


def multihot_encode(series: pd.Series) -> pd.DataFrame:
    all_categories = set()
    series.map(
        lambda categories: [
            all_categories.add(categories)
            for categories in (categories if isinstance(categories, list) else [])
        ]
    )
    all_categories = list(sorted(all_categories))
    records = series.map(
        lambda categories: {
            f"{series.name}: {unique_category}": unique_category
            in (categories if isinstance(categories, list) else [])
            for unique_category in all_categories
        }
    ).tolist()
    dataframe = pd.DataFrame.from_records(records)
    return dataframe[list(sorted(dataframe.columns))].astype(float)


def lambda_onehot_encode(series: pd.Series, fn: Callable) -> pd.DataFrame:
    series = series.astype(float)
    max_value = series.max()
    max_lambda_value = fn(max_value)
    return onehot_encode(series.map(lambda value: min(fn(value), max_lambda_value)))


def sqrt_onehot_encode(series: pd.Series) -> pd.DataFrame:
    return lambda_onehot_encode(series, fn=lambda x: int(x**0.5))


def log_onehot_encode(series: pd.DataFrame) -> pd.DataFrame:
    return lambda_onehot_encode(series, fn=lambda x: int(math.log(x)))


def z_score_scale(
    series: pd.Series,
    clip: bool = False,
    lower: int = None,
    higher: int = None,
) -> pd.DataFrame:
    if clip:
        series = series.clip(lower=lower, higher=higher)
    mean = series.mean()
    std = series.std()
    transformed = (series - mean) / std
    return transformed.to_frame()


def min_max_scale(
    series: pd.Series,
    clip: bool = False,
    lower: int = None,
    higher: int = None,
) -> pd.DataFrame:
    if clip:
        series = series.clip(lower=lower, higher=higher)
    _max = series.max()
    _min = series.min()
    transformed = (series - _min) / (_max - _min)
    return transformed.to_frame()


def concat_encodings(dataframes: Sequence[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat([e.reset_index(drop=True) for e in dataframes], axis=1)
