"""Offline checks for the current single-label package; no model training or Drive access."""

import subprocess
import sys
from pathlib import Path

import pytest

from vietnamese_topic_classification import __version__
from vietnamese_topic_classification.config import get_domain_labels, load_domain_config
from vietnamese_topic_classification.config.settings import DataLakeConfig


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


def test_datalake_config_from_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_LAKE_ROOT", str(tmp_path))
    assert DataLakeConfig.from_env().root == tmp_path


def test_datalake_config_from_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DATA_LAKE_ROOT", raising=False)
    (tmp_path / ".env").write_text(f"DATA_LAKE_ROOT={tmp_path}\n", encoding="utf-8")

    assert DataLakeConfig.from_env().root == tmp_path
    monkeypatch.delenv("DATA_LAKE_ROOT", raising=False)


def test_environment_overrides_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("DATA_LAKE_ROOT=/missing-folder\n", encoding="utf-8")
    monkeypatch.setenv("DATA_LAKE_ROOT", str(tmp_path))

    assert DataLakeConfig.from_env().root == tmp_path


def test_datalake_config_requires_variable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DATA_LAKE_ROOT", raising=False)
    with pytest.raises(ValueError, match="DATA_LAKE_ROOT"):
        DataLakeConfig.from_env()


def test_datalake_config_requires_existing_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_LAKE_ROOT", str(tmp_path / "does-not-exist"))
    with pytest.raises(FileNotFoundError, match="Data Lake directory"):
        DataLakeConfig.from_env()


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
