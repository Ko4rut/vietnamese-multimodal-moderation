"""Versioned 25-domain single-label dataset configuration."""

from importlib.resources import files
from typing import Any

import yaml


def load_domain_config() -> dict[str, Any]:
    """Read the packaged VTSNLP domain configuration without network access."""
    path = files("vietnamese_topic_classification").joinpath("config", "domains_v1.yaml")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("task") != "single_label_multiclass":
        raise ValueError("Expected a single-label multiclass domain configuration")
    labels = config.get("domains")
    if (
        not isinstance(labels, list)
        or len(labels) != 25
        or any(not isinstance(label, str) or not label for label in labels)
        or len(set(labels)) != 25
    ):
        raise ValueError("Domain configuration must contain exactly 25 unique labels")
    return config


def get_domain_labels() -> tuple[str, ...]:
    """Return the canonical order for label-to-index mapping (do not reorder)."""
    return tuple(load_domain_config()["domains"])
