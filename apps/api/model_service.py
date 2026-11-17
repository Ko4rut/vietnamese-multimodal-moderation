"""Load a trusted, fitted sklearn experiment and provide single-text inference.

The model artifact is always supplied by the operator via VTC_MODEL_DIR.
Never accept serialized model files from an HTTP request: joblib is pickle-based.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from vietnamese_topic_classification.config import get_domain_labels


class ModelUnavailable(RuntimeError):
    """Raised if no usable experiment artifact has been configured."""


@dataclass(frozen=True)
class LoadedModel:
    pipeline: Any
    experiment_name: str
    classifier: str
    validation_macro_f1: float | None
    classes: tuple[int, ...]


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object")
    return data


@lru_cache(maxsize=4)
def _load_from_directory(directory: str) -> LoadedModel:
    root = Path(directory)
    metadata = _read_json(root / "metadata.json")
    config = _read_json(root / "config.json")
    classifier_config = config.get("classifier")
    if not isinstance(classifier_config, dict):
        raise ValueError("Missing classifier config")
    classifier = classifier_config.get("kind")
    if classifier not in ("logistic_regression", "linear_svc"):
        raise ValueError("Unsupported classifier")
    pipeline = joblib.load(root / "model.joblib")
    if not callable(getattr(pipeline, "predict", None)):
        raise ValueError("Model has no predict method")
    raw_classes = getattr(pipeline, "classes_", None)
    if raw_classes is None:
        raise ValueError("Model has no fitted classes")
    classes = tuple(int(label_id) for label_id in raw_classes)
    labels = get_domain_labels()
    if (
        len(classes) < 2
        or len(set(classes)) != len(classes)
        or any(label_id < 0 or label_id >= len(labels) for label_id in classes)
    ):
        raise ValueError("Model classes do not match the domain configuration")
    if classifier == "logistic_regression" and not callable(
        getattr(pipeline, "predict_proba", None)
    ):
        raise ValueError("Logistic Regression model lacks probabilities")
    if classifier == "linear_svc" and not callable(
        getattr(pipeline, "decision_function", None)
    ):
        raise ValueError("LinearSVC model lacks decision scores")
    name = metadata.get("experiment_name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Missing experiment name")
    metrics = metadata.get("validation_metrics", {})
    score = metrics.get("macro_f1") if isinstance(metrics, dict) else None
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        score = None
    elif not math.isfinite(float(score)):
        score = None
    return LoadedModel(
        pipeline=pipeline,
        experiment_name=name,
        classifier=classifier,
        validation_macro_f1=float(score) if score is not None else None,
        classes=classes,
    )


def get_loaded_model() -> LoadedModel:
    configured_path = os.environ.get("VTC_MODEL_DIR", "").strip()
    if not configured_path:
        raise ModelUnavailable(
            "Chưa cấu hình VTC_MODEL_DIR trỏ tới thư mục experiment đã train."
        )
    root = Path(configured_path).expanduser()
    if not root.is_dir() or not all(
        (root / filename).is_file()
        for filename in ("model.joblib", "metadata.json", "config.json")
    ):
        raise ModelUnavailable(
            "Không tìm thấy model.joblib, metadata.json và config.json trong VTC_MODEL_DIR."
        )
    try:
        return _load_from_directory(str(root.resolve()))
    except Exception as exc:
        raise ModelUnavailable(
            "Không thể tải model. Kiểm tra artifact và phiên bản scikit-learn."
        ) from exc


def predict_text(model: LoadedModel, text: str, top_k: int = 5) -> dict[str, Any]:
    """Return real model scores; SVC margins are *never* called probabilities."""
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
        "model_name": model.experiment_name,
        "label_id": predicted_id,
        "label": labels[predicted_id],
        "score_kind": score_kind,
        "top_predictions": [
            {"label_id": label_id, "label": labels[label_id], "score": score}
            for label_id, score in ranked[:top_k]
        ],
    }
