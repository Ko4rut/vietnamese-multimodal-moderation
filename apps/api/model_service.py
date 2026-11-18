"""Discover and serve trusted, previously trained TF-IDF baseline artifacts.

Only operator-configured directories are scanned. The HTTP client may select
a *registered model ID*, never a filesystem path or uploaded pickle.
"""

from __future__ import annotations

import json
import math
import os
import re
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from vietnamese_topic_classification.config import get_domain_labels

ClassifierName = Literal["logistic_regression", "linear_svc"]
MODEL_FILES = ("model.joblib", "metadata.json", "config.json")
MODEL_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


class ModelUnavailable(RuntimeError):
    """A configured model cannot be served (missing or incompatible artifact)."""


class ModelNotFound(ModelUnavailable):
    """The requested model ID is not registered."""


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    model_id: str
    directory: Path
    classifier: ClassifierName
    validation_macro_f1: float | None
    sklearn_version: str | None

    def as_response(self) -> dict[str, object]:
        return {
            "id": self.model_id,
            "model_name": self.model_id,
            "classifier": self.classifier,
            "validation_macro_f1": self.validation_macro_f1,
            "sklearn_version": self.sklearn_version,
        }


@dataclass(frozen=True, slots=True)
class LoadedModel:
    descriptor: ModelDescriptor
    pipeline: Pipeline
    classes: tuple[int, ...]

    @property
    def experiment_name(self) -> str:
        return self.descriptor.model_id

    @property
    def classifier(self) -> ClassifierName:
        return self.descriptor.classifier

    @property
    def validation_macro_f1(self) -> float | None:
        return self.descriptor.validation_macro_f1


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object")
    return data


def _discover_directories() -> list[Path]:
    """Locate complete experiment folders under VTC_MODELS_ROOT or VTC_MODEL_DIR."""
    directories: set[Path] = set()
    root_value = os.getenv("VTC_MODELS_ROOT", "").strip()
    if root_value:
        root = Path(root_value).expanduser().resolve()
        if not root.is_dir():
            raise ModelUnavailable("VTC_MODELS_ROOT không tồn tại hoặc không phải thư mục.")
        # This root should contain experiment folders, not the entire Data Lake.
        for metadata in root.rglob("metadata.json"):
            directory = metadata.parent.resolve()
            if not directory.is_relative_to(root):
                continue
            if all((directory / filename).is_file() for filename in MODEL_FILES):
                directories.add(directory)

    single_value = os.getenv("VTC_MODEL_DIR", "").strip()
    if single_value:
        directory = Path(single_value).expanduser().resolve()
        if not directory.is_dir() or not all(
            (directory / filename).is_file() for filename in MODEL_FILES
        ):
            raise ModelUnavailable(
                "VTC_MODEL_DIR cần chứa model.joblib, metadata.json và config.json."
            )
        directories.add(directory)
    return sorted(directories)


def _descriptor_for(directory: Path) -> ModelDescriptor:
    try:
        metadata = _read_json(directory / "metadata.json")
        config = _read_json(directory / "config.json")
        name = metadata.get("experiment_name")
        classifier_config = config.get("classifier")
        if not isinstance(name, str) or not MODEL_ID_PATTERN.fullmatch(name):
            raise ValueError("Invalid experiment name")
        if config.get("name") is not None and config["name"] != name:
            raise ValueError("Experiment identity mismatch between metadata and config")
        if not isinstance(classifier_config, dict):
            raise ValueError("Missing classifier config")
        classifier = classifier_config.get("kind")
        if classifier not in ("logistic_regression", "linear_svc"):
            raise ValueError("Unsupported classifier")
        features = config.get("features")
        if features is not None and (
            not isinstance(features, dict) or features.get("kind") != "tfidf"
        ):
            raise ValueError("Only TF-IDF baselines are supported")
        sklearn_version = metadata.get("sklearn_version")
        if sklearn_version is not None and not isinstance(sklearn_version, str):
            raise ValueError("Invalid scikit-learn version")
        metrics = metadata.get("validation_metrics")
        raw_f1 = metrics.get("macro_f1") if isinstance(metrics, dict) else None
        f1 = (
            float(raw_f1)
            if isinstance(raw_f1, (float, int))
            and not isinstance(raw_f1, bool)
            and math.isfinite(float(raw_f1))
            and 0 <= raw_f1 <= 1
            else None
        )
        return ModelDescriptor(
            model_id=name,
            directory=directory,
            classifier=classifier,
            validation_macro_f1=f1,
            sklearn_version=sklearn_version,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ModelUnavailable(
            f"Artifact không hợp lệ trong experiment '{directory.name}'."
        ) from exc


def list_models() -> list[ModelDescriptor]:
    """Read lightweight metadata; never unpickle models during discovery."""
    found: dict[str, ModelDescriptor] = {}
    for directory in _discover_directories():
        descriptor = _descriptor_for(directory)
        if descriptor.model_id in found and found[descriptor.model_id].directory != directory:
            raise ModelUnavailable(
                f"Trùng tên experiment '{descriptor.model_id}'. Hãy giữ mỗi tên một bản."
            )
        found[descriptor.model_id] = descriptor
    return [found[key] for key in sorted(found)]


def default_model_id(descriptors: list[ModelDescriptor] | None = None) -> str | None:
    catalog = descriptors if descriptors is not None else list_models()
    configured = os.getenv("VTC_DEFAULT_MODEL", "").strip()
    names = {item.model_id for item in catalog}
    if configured and configured not in names:
        raise ModelNotFound(f"VTC_DEFAULT_MODEL '{configured}' không có trong registry.")
    return configured or (catalog[0].model_id if catalog else None)


def _resolve_descriptor(model_id: str | None) -> ModelDescriptor:
    descriptors = list_models()
    selected = model_id if model_id is not None else default_model_id(descriptors)
    if selected is None:
        raise ModelUnavailable(
            "Chưa có artifact. Hãy cấu hình VTC_MODEL_DIR hoặc VTC_MODELS_ROOT."
        )
    for descriptor in descriptors:
        if descriptor.model_id == selected:
            return descriptor
    raise ModelNotFound("Không tìm thấy model_id đã yêu cầu.")


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
    descriptor = _resolve_descriptor(model_id)
    _check_sklearn_version(descriptor)
    try:
        pipeline, classes = _load_pipeline(str(descriptor.directory), descriptor.classifier)
        return LoadedModel(descriptor=descriptor, pipeline=pipeline, classes=classes)
    except (OSError, ValueError, TypeError, AttributeError, EOFError, ImportError) as exc:
        raise ModelUnavailable(
            f"Không thể load model '{descriptor.model_id}'. Kiểm tra artifact và dependencies."
        ) from exc


def predict_text(model: LoadedModel, text: str, top_k: int = 5) -> dict[str, object]:
    """Return genuine model rankings; a LinearSVC margin is not a probability."""
    if not (1 <= top_k):
        raise ValueError("top_k is out of range")
    top_k = min(top_k, len(model.classes))
    labels = get_domain_labels()
    predicted_id = int(model.pipeline.predict([text])[0])
    if predicted_id not in model.classes:
        raise ValueError("Prediction outside fitted classes")

    if model.classifier == "logistic_regression":
        values = [float(x) for x in model.pipeline.predict_proba([text])[0]]
        score_kind = "probability"
    else:
        raw = model.pipeline.decision_function([text])[0]
        if len(model.classes) == 2:
            signed_margin = float(raw)
            values = [-signed_margin, signed_margin]
        else:
            values = [float(x) for x in raw]
        score_kind = "decision_margin"

    if len(values) != len(model.classes) or not all(math.isfinite(x) for x in values):
        raise ValueError("Invalid inference scores")
    ranked = sorted(zip(model.classes, values), key=lambda pair: pair[1], reverse=True)
    return {
        "model_id": model.descriptor.model_id,
        "model_name": model.experiment_name,
        "label_id": predicted_id,
        "label": labels[predicted_id],
        "score_kind": score_kind,
        "top_predictions": [
            {"label_id": label_id, "label": labels[label_id], "score": score}
            for label_id, score in ranked[:top_k]
        ],
    }
