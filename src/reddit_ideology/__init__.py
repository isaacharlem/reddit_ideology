"""
reddit_ideology

Provides a high-level API for semantic analysis of Reddit discourse.
"""

from .config import load_config
from .data_loader import DataLoader
from .preprocessing import Preprocessor
from .embedding import EmbeddingModel
from .topic_model import EmbeddingClusterTopicModel as TopicModel
from .metrics import MetricsCalculator
from .visualize import Visualizer
from .cli import main

__all__ = [
    "load_config",
    "DataLoader",
    "Preprocessor",
    "EmbeddingModel",
    "TopicModel",
    "MetricsCalculator",
    "Visualizer",
    "main",
]
