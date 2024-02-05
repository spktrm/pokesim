import pandas as pd

from typing import Any, Dict, List


def to_id(string: str) -> str:
    return "".join(c for c in string if c.isalnum()).lower()


def get_df(data: List[Dict[str, Any]]):
    df = pd.json_normalize(data)
    for mask_fn in [
        lambda: df["isNonstandard"].map(lambda x: x is None),
        lambda: (df["tier"] != "Illegal"),
    ]:
        try:
            mask = mask_fn()
            df = df[mask]
        except Exception:
            # traceback.print_exc()
            return df
    return df
