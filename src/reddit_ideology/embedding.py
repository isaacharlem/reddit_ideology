import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class EmbeddingModel:
    def __init__(
        self, model_name: str, batch_size: int, device: str, cache_dir: str
    ):
        # Determine device
        if device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            print("CUDA not available, using CPU")
        elif device == "mps" and not getattr(torch.backends, "mps", None):
            self.device = "cpu"
            print("MPS not available, using CPU")
        else:
            self.device = device
        self.model = SentenceTransformer(model_name, device=self.device)
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def embed(self, texts: list, name: str) -> np.ndarray:
        cache_path = os.path.join(self.cache_dir, f"{name}_embeddings.npy")
        if os.path.exists(cache_path):
            return np.load(cache_path)
        embeddings = []
        for i in tqdm(
            range(0, len(texts), self.batch_size), desc=f"Embedding {name}"
        ):
            batch = texts[i : i + self.batch_size]
            emb = self.model.encode(
                batch, convert_to_numpy=True, show_progress_bar=False
            )
            embeddings.append(emb)
        embeddings = np.vstack(embeddings)
        np.save(cache_path, embeddings)
        return embeddings
