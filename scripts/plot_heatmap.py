from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
import torch

import numpy as np
import plotly.express as px

from sklearn.metrics import pairwise_distances

from pokesim.data import ABILITIES_STOI, ITEMS_STOI, MOVES_STOI, SPECIES_STOI
from pokesim.embeddings.moves import construct_moves_encoding
from pokesim.embeddings.species import construct_species_encoding
from pokesim.nn.model import Model
from pokesim.utils import get_most_recent_file


def main(gen: int = 3):
    data_npy = np.load(f"src/data/gen{gen}/moves.npy")
    data = data_npy

    fpath = get_most_recent_file("ckpts")
    data_torch = torch.load(fpath)["params"]

    model = Model(32, 128, 8, True, True)
    model.load_state_dict(data_torch)

    data = (
        model.policy_head.moves_onehot(
            torch.arange(data_npy.shape[0], dtype=torch.long)
        )
        .detach()
        .numpy()
    )

    indices = []
    names = []

    for key, value in MOVES_STOI.items():
        if data_npy[value].sum() != 0:
            names.append(key)
            indices.append(value)

    data = data[np.array(indices)]

    data = PCA(128).fit_transform(data)
    data = StandardScaler().fit_transform(data)

    pairwise = 1 - pairwise_distances(data, metric="cosine")

    fig = px.imshow(pairwise, text_auto=True, x=names, y=names)
    fig.show()


if __name__ == "__main__":
    main()
