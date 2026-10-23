"""Versioned application configuration and taxonomy resources."""

from importlib.resources import files

import yaml


def load_taxonomy() -> dict:
    """Load bundled taxonomy regardless of the current working directory."""
    resource = files("moderation").joinpath("config/policies/mcst_v1.yaml")
    return yaml.safe_load(resource.read_text(encoding="utf-8"))
