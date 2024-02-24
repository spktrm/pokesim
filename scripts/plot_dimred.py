import numpy as np

import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.manifold import TSNE
from pokesim.data import ABILITIES_STOI, ITEMS_STOI, MOVES_STOI, SPECIES_STOI
from pokesim.embeddings.helpers import Encoder, pred

from pokesim.embeddings.abilities import construct_abilities_encoding
from pokesim.embeddings.species import construct_species_encoding
from pokesim.embeddings.moves import construct_moves_encoding


def perform_pca(data: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Perform PCA on the given data and return the transformed data and explained variance.

    Parameters:
    - data: np.ndarray, the input data array of shape (n_samples, n_features).
    - n_components: int, the number of principal components to compute.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: A tuple containing the transformed data and the explained variance ratio.
    """
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data)
    print(pca.explained_variance_ratio_.sum())
    return transformed_data


def perform_tsne(data: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Perform PCA on the given data and return the transformed data and explained variance.

    Parameters:
    - data: np.ndarray, the input data array of shape (n_samples, n_features).
    - n_components: int, the number of principal components to compute.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: A tuple containing the transformed data and the explained variance ratio.
    """
    tsne = TSNE(n_components=n_components)

    transformed_data = tsne.fit_transform(data)
    return transformed_data


def plot_pca_2d(data: np.ndarray, title: str = "2D PCA Plot", **kwargs) -> None:
    """
    Plot the first two principal components of the data in 2D.

    Parameters:
    - data: np.ndarray, the transformed data array of shape (n_samples, 2).
    - title: str, the title of the plot.
    """
    fig = px.scatter(x=data[..., 0], y=data[..., 1], title=title, **kwargs)
    fig.show()


def plot_pca_3d(data: np.ndarray, title: str = "3D PCA Plot", **kwargs) -> None:
    """
    Plot the first three principal components of the data in 3D.

    Parameters:
    - data: np.ndarray, the transformed data array of shape (n_samples, 3).
    - title: str, the title of the plot.
    """
    fig = px.scatter(
        x=data[..., 0], y=data[..., 1], z=data[..., 2], title=title, **kwargs
    )
    fig.show()


def main(gen: int = 3, ndims: int = 2):
    data = np.load(f"src/data/gen{gen}/species.npy")

    indices = []
    names = []

    for key, value in SPECIES_STOI.items():
        if data[value].sum() != 0:
            names.append(key)
            indices.append(value)

    transformed_data = data[np.array(indices)]

    transformed_data = perform_pca(
        transformed_data, n_components=min(data.shape[-1], 64)
    )
    labels = KMeans(n_clusters=16).fit_predict(transformed_data)

    transformed_data = transformed_data[..., :ndims]

    if ndims == 2:
        plot_pca_2d(transformed_data, text=names, color=labels.astype(str))
    elif ndims == 3:
        plot_pca_3d(transformed_data, text=names, color=labels.astype(str))
    return


# Example usage:
if __name__ == "__main__":
    main()
