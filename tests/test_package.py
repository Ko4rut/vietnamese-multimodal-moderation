"""Offline checks for the current single-label package; no model training or Drive access."""

import subprocess
import sys
from pathlib import Path

import pytest

from vietnamese_topic_classification import __version__
from vietnamese_topic_classification.config import get_domain_labels, load_domain_config
from vietnamese_topic_classification.config.settings import get_settings


def test_domain_config_matches_current_task() -> None:
    cfg = load_domain_config()
    assert cfg["task"] == "single_label_multiclass"
    assert cfg["num_classes"] == 25
    assert cfg["dataset"]["label_column"] == "domain"
    assert cfg["dataset"]["text_column"] == "text"
    assert cfg["dataset"]["revision"] == "b81fcce58945970117a1b56d50ec81be2628a5c3"
    labels = get_domain_labels()
    assert len(labels) == len(set(labels)) == 25
    assert "Online_Communities" in labels
    assert "Arts_and_Entertainment" in labels


def test_config_loads_from_installed_package() -> None:
    labels = get_domain_labels()
    assert labels[0] == "Arts_and_Entertainment"
    assert labels[-1] == "Travel_and_Transportation"


def test_optional_data_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VTC_DATA_ROOT", raising=False)
    assert get_settings().data_root is None
    monkeypatch.setenv("VTC_DATA_ROOT", "/tmp/colab-data")
    assert get_settings().data_root == Path("/tmp/colab-data")


def test_cli_lists_all_domains() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "vietnamese_topic_classification", "--list-domains"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == list(get_domain_labels())


def test_version_is_consistent() -> None:
    assert __version__ == "0.1.0"
