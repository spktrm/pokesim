import numpy as np

from tabulate import tabulate

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import pairwise_distances
import torch

from pokesim.data import MOVES_STOI, SPECIES_STOI
from pokesim.embeddings.helpers import to_id
from pokesim.nn.model import Model
from pokesim.utils import get_most_recent_file


def main(gen: int = 3, max_similarity: float = 0.5):
    data_npy = np.load(f"src/data/gen{gen}/species.npy")

    fpath = get_most_recent_file("ckpts")
    data_torch = torch.load(fpath)["params"]

    model = Model(32, 128, 8, True, True)
    model.load_state_dict(data_torch)

    data = (
        model.encoder.species_onehot(torch.arange(data_npy.shape[0], dtype=torch.long))
        .detach()
        .numpy()
    )

    indices = []
    names = []

    itos = {}
    stoi = {}

    n = 0
    for key, value in SPECIES_STOI.items():
        if data_npy[value].sum() != 0:
            names.append(key)
            indices.append(value)

            itos[n] = key
            stoi[key] = n
            n += 1

    data = data[np.array(indices)]

    # data = PCA(128).fit_transform(data)
    # data = StandardScaler().fit_transform(data)

    pairwise = 1 - pairwise_distances(data, metric="cosine")

    while True:
        try:
            move = to_id(input().strip())
            index = stoi.get(move)
            if index is None:
                continue

            similarity = pairwise[index]

            table_values = []
            for top_ind in np.argsort(similarity)[::-1]:
                sim_val = similarity[top_ind]
                if sim_val > max_similarity and sim_val != 1:
                    table_values.append([itos[top_ind], f"{sim_val:.3f}"])
            print(tabulate(table_values))

        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    main()
