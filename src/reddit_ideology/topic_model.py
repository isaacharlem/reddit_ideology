import os
import pickle
from typing import List, Union

import numpy as np
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer


class EmbeddingClusterTopicModel:
    """
    A wrapper around BERTopic using KeyBERT-inspired topic representation.

    Replaces UMAP+HDBSCAN with BERTopic's integrated pipeline.
    """
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        vectorizer_params: dict = None,
        nr_topics: Union[str, int] = "auto",
        top_n_words: int = 10,
        min_topic_size: int = 20,
        cache_dir: str = "cache"
    ):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # 1. SentenceTransformer for embeddings
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # 2. CountVectorizer to drop filler terms
        default_vec = {
            "ngram_range": (1, 2),
            "stop_words": "english",
            "min_df": 10,
            "max_df": 0.90
        }
        vec_args = vectorizer_params or default_vec
        self.vectorizer_model = CountVectorizer(**vec_args)

        # 3. KeyBERT-inspired representation from BERTopic
        self.representation_model = KeyBERTInspired(
            top_n_words=top_n_words
        )

        # 4. BERTopic instantiation
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            vectorizer_model=self.vectorizer_model,
            representation_model=self.representation_model,
            nr_topics=nr_topics,
            top_n_words=top_n_words,
            min_topic_size=min_topic_size,
            verbose=True
        )

    def fit(self, docs: List[str], name: str) -> np.ndarray:
        """
        Fit BERTopic on a list of documents (cleaned text), cache and return topic labels.
        """
        cache_path = os.path.join(self.cache_dir, f"{name}_topics.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                labels = pickle.load(f)
            return labels

        # Fit & transform
        topics, _ = self.topic_model.fit_transform(docs)
        with open(cache_path, 'wb') as f:
            pickle.dump(topics, f)
        return np.array(topics)

    def get_topic_info(self):
        """Return a DataFrame of topic IDs and their top keywords."""
        return self.topic_model.get_topic_info()

    def get_topic(self, topic_id: int):
        """Return keywords for a specific topic id."""
        return self.topic_model.get_topic(topic_id)

    def save(self, path: str):
        """Save the BERTopic model to disk."""
        self.topic_model.save(path)

    def load(self, path: str):
        """Load a saved BERTopic model from disk."""
        self.topic_model = BERTopic.load(path)
