"""Feature and estimator construction boundary.

Phase 1 will return a scikit-learn Pipeline containing both the TF-IDF
transformer and classifier, so feature fitting happens on train only.
The factory can later delegate to a provider/engine abstraction if needed.
"""

from __future__ import annotations

from .config import ExperimentConfig


def build_pipeline(config: ExperimentConfig) -> object:
    """Construct an unfitted model from typed settings; never fit here."""
    raise NotImplementedError("Skeleton only: implement the model pipeline factory")
