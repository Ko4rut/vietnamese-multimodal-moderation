"""Validate and load a fitted TF-IDF sklearn pipeline once per process."""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from vietnamese_topic_classification.config import get_domain_labels

from .registry import resolve_descriptor
from .types import ClassifierName, LoadedModel, ModelDescriptor, ModelUnavailable

def _check_sklearn_version(descriptor: ModelDescriptor) -> None:
    stored = descriptor.sklearn_version
    if stored is None:
        warnings.warn(
            "Artifact has no sklearn_version; compatibility cannot be verified.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    if stored.split(".")[:2] != sklearn.__version__.split(".")[:2]:
        raise ModelUnavailable(
            "Phiên bản scikit-learn không tương thích: model được train với "
            f"{stored}, server đang dùng {sklearn.__version__}. "
            "Hãy cài đúng major.minor version trước khi load."
        )


@lru_cache(maxsize=8)
def _load_pipeline(directory: str, classifier_name: ClassifierName) -> tuple[Pipeline, tuple[int, ...]]:
    """Load only once per process; restart the server to replace an artifact."""
    pipeline = joblib.load(Path(directory) / "model.joblib")
    if not isinstance(pipeline, Pipeline):
        raise ValueError("Expected an sklearn Pipeline")
    if not isinstance(pipeline.named_steps.get("tfidf"), TfidfVectorizer):
        raise ValueError("Model must contain a fitted TF-IDF transformer")
    estimator = pipeline.named_steps.get("classifier")
    expected_type = (
        LogisticRegression if classifier_name == "logistic_regression" else LinearSVC
    )
    if not isinstance(estimator, expected_type):
        raise ValueError("Classifier type does not match config.json")
    if not callable(getattr(pipeline, "predict", None)):
        raise ValueError("Missing predict method")
    raw_classes = getattr(pipeline, "classes_", None)
    if raw_classes is None:
        raise ValueError("Unfitted classifier")
    classes = tuple(int(item) for item in raw_classes)
    labels = get_domain_labels()
    if (
        len(classes) < 2
        or len(classes) != len(set(classes))
        or any(item < 0 or item >= len(labels) for item in classes)
    ):
        raise ValueError("Classes do not match canonical domain IDs")
    return pipeline, classes


def get_loaded_model(model_id: str | None = None) -> LoadedModel:
    descriptor = resolve_descriptor(model_id)
    _check_sklearn_version(descriptor)
    try:
        pipeline, classes = _load_pipeline(str(descriptor.directory), descriptor.classifier)
        return LoadedModel(descriptor=descriptor, pipeline=pipeline, classes=classes)
    except Exception as exc:
        # joblib/pickle can raise many errors for corrupt or incompatible trusted files.
        # Expose a controlled 503 to clients; never return internal traceback details.
        raise ModelUnavailable(
            f"Không thể load model '{descriptor.model_id}'. Kiểm tra artifact và dependencies."
        ) from exc
