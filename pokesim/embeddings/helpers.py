import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from typing import Any, Callable, Dict, List, Sequence, TypedDict

from pokesim.nn.modules import MLP, _layer_init


class Protocol(TypedDict):
    feature: str
    feature_fn: Callable[[str], bool]
    func: Callable[[pd.Series], pd.DataFrame]


def to_id(string: str) -> str:
    return "".join(c for c in string if c.isalnum()).lower()


def get_df(data: List[Dict[str, Any]]):
    data = [
        d
        for d in data
        if (d.get("isNonstandard") is None) and (d.get("tier") != "Illegal")
    ]

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
            # return df
            pass

    cols_to_drop = []
    for column in df.columns:
        if len(df[column].map(lambda x: json.dumps(x)).unique()) <= 1:
            cols_to_drop.append(column)
    df = df.drop(cols_to_drop, axis=1)
    df.index = df["id"]
    return df


class Encoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int = None,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        self._input_size = input_size
        self._hidden_sizes = hidden_sizes
        self._output_size = output_size or input_size

        self.encoder = MLP(
            [self._input_size, *self._hidden_sizes], use_layer_norm=use_layer_norm
        )
        self.bottleneck = _layer_init(
            nn.Linear(self._hidden_sizes[-1], self._hidden_sizes[-1])
        )
        self.decoder = MLP(
            [*(self._hidden_sizes[::-1]), self._output_size],
            use_layer_norm=use_layer_norm,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        mean = encoded.mean(-1, keepdims=True)
        std = encoded.std(-1, keepdims=True)
        return (encoded - mean) / std

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.bottleneck(F.relu(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def pred(
    X: np.ndarray,
    y: np.ndarray,
    model: Encoder,
    num_epochs: int = 10,
    lr: float = 1e-4,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = torch.from_numpy(X).to(device)
    y = torch.from_numpy(y).to(device)
    model = model.to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=lr)

    for i in range(num_epochs):
        optimizer.zero_grad()
        out = model(X)
        loss = ((X - out) ** 2).mean(-1).mean()
        print(f"{i}\t{loss.item():.5f}")
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        return model.encode(X).detach().cpu().numpy()
