import torch
import numpy as np
import plotly.express as px

from sklearn.metrics import pairwise_distances

from pokesim.data import ABILITIES_STOI, ITEMS_STOI, MOVES_STOI, SPECIES_STOI


def main(gen: int = 3):
    data = np.load(f"src/data/gen{gen}/species.npy")
    data1 = (
        torch.load(f"ckpts/023594.pt")["params"]["encoder.species_onehot.weight"]
        .cpu()
        .numpy()
    )

    indices = []
    names = []

    for key, value in SPECIES_STOI.items():
        if data[value].sum() != 0:
            names.append(key)
            indices.append(value)

    data = data1[np.array(indices)]

    pairwise = 1 - pairwise_distances(data, metric="cosine")

    fig = px.imshow(pairwise, text_auto=True, x=names, y=names)
    fig.show()


if __name__ == "__main__":
    main()
