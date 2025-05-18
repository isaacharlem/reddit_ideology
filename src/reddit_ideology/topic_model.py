import os
import pickle
import numpy as np
import hdbscan
import umap

class EmbeddingClusterTopicModel:
    def __init__(
        self,
        umap_neighbors: int,
        umap_min_dist: float,
        hdbscan_min_cluster_size: int,
        cache_dir: str
    ):
        self.umap_neighbors = umap_neighbors
        self.umap_min_dist = umap_min_dist
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.reducer = umap.UMAP(
            n_neighbors=self.umap_neighbors,
            min_dist=self.umap_min_dist,
            metric='cosine'
        )
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.hdbscan_min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom'
        )

    def fit(self, embeddings: np.ndarray, name: str) -> np.ndarray:
        cache_path = os.path.join(self.cache_dir, f"{name}_clusters.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                labels = pickle.load(f)
            return labels
        reduced = self.reducer.fit_transform(embeddings)
        labels = self.clusterer.fit_predict(reduced)
        with open(cache_path, 'wb') as f:
            pickle.dump(labels, f)
        return labels