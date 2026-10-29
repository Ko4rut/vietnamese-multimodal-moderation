"""Training scaffolding for reproducible Vietnamese topic classification.

Importing this package does not load a dataset or train a model.
"""

from .config import ClassifierConfig, ExperimentConfig, FeatureConfig
from .model import DatasetSplits, ExperimentResult, TextSplit

__all__ = [
    "ClassifierConfig",
    "DatasetSplits",
    "ExperimentConfig",
    "ExperimentResult",
    "FeatureConfig",
    "TextSplit",
]
