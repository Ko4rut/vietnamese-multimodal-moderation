"""Lightweight discovery of trusted experiment directories; no joblib loading."""

from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

from .types import MODEL_FILES, ModelDescriptor, ModelNotFound, ModelUnavailable

MODEL_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")

LOGGER = logging.getLogger(__name__)


def _default_models_root() -> Path:
    """Resolve project-local models/ independently of the working directory."""
    return Path(__file__).resolve().parents[3] / "models"


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object")
    return data


def _discover_directories() -> list[Path]:
    """Discover project-local experiments, unless an explicit path overrides them."""
    directories: set[Path] = set()
    root_value = os.getenv("VTC_MODELS_ROOT", "").strip()
    single_value = os.getenv("VTC_MODEL_DIR", "").strip()

    root = (
        Path(root_value).expanduser().resolve()
        if root_value
        else (_default_models_root() if not single_value else None)
    )
    if root is not None:
        if not root.is_dir():
            if root_value:
                raise ModelUnavailable("VTC_MODELS_ROOT không tồn tại hoặc không phải thư mục.")
            # An empty fresh checkout is valid; models/ can be added later.
        else:
            resolved_root = root.resolve()
            for metadata in root.rglob("metadata.json"):
                directory = metadata.parent.resolve()
                if not directory.is_relative_to(resolved_root):
                    continue
                if all((directory / filename).is_file() for filename in MODEL_FILES):
                    directories.add(directory)

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
    """Read metadata on every request; never unpickle during discovery."""
    found: dict[str, ModelDescriptor] = {}
    explicit = bool(
        os.getenv("VTC_MODELS_ROOT", "").strip()
        or os.getenv("VTC_MODEL_DIR", "").strip()
    )
    for directory in _discover_directories():
        try:
            descriptor = _descriptor_for(directory)
        except ModelUnavailable:
            if explicit:
                raise
            # A malformed local artifact must not hide other valid experiments.
            LOGGER.warning("Bỏ qua artifact không hợp lệ: %s", directory, exc_info=True)
            continue
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


def resolve_descriptor(model_id: str | None) -> ModelDescriptor:
    descriptors = list_models()
    selected = model_id if model_id is not None else default_model_id(descriptors)
    if selected is None:
        raise ModelUnavailable(
            "Chưa có artifact. Đặt experiment vào models/ hoặc cấu hình VTC_MODELS_ROOT."
        )
    for descriptor in descriptors:
        if descriptor.model_id == selected:
            return descriptor
    raise ModelNotFound("Không tìm thấy model_id đã yêu cầu.")
