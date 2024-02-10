import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler

from tqdm import tqdm
from contextlib import redirect_stdout

from pokesim.embeddings.abilities import construct_abilities_encoding
from pokesim.embeddings.helpers import Encoder, pred
from pokesim.embeddings.items import construct_items_encoding
from pokesim.embeddings.moves import construct_moves_encoding
from pokesim.embeddings.species import construct_species_encoding


func_mapping = {
    "species": construct_species_encoding,
    "abilities": construct_abilities_encoding,
    "moves": construct_moves_encoding,
    "items": construct_items_encoding,
}

NPC = 128


def main():
    bar1 = tqdm(range(3, 4), position=0)
    for gen in bar1:
        bar1.set_description(f"gen{gen}")

        bar2 = tqdm(func_mapping.items(), position=1, leave=False)
        for name, func in bar2:
            bar2.set_description(name)
            with redirect_stdout(None):
                encoding = func(gen)
            valid_indices = encoding.sum(-1) != 0

            # new = np.zeros((encoding.shape[0], NPC))
            new = np.zeros(encoding.shape)

            arr = encoding[valid_indices].copy()
            # arr = pred(
            #     arr,
            #     np.arange(encoding.shape[0]),
            #     Encoder(
            #         encoding.shape[-1],
            #         [256, NPC],
            #         # use_layer_norm=False,
            #     ),
            #     10000,
            # )
            # arr = StandardScaler().fit_transform(arr)

            new[valid_indices] = arr
            np.save(f"src/data/gen{gen}/{name}.npy", new)


if __name__ == "__main__":
    main()
