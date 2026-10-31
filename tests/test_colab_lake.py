"""Offline tests for collaborator-specific Google Drive path resolution."""

from pathlib import Path

import pytest

from vietnamese_topic_classification.data.colab_lake import (
    DEFAULT_LAKE_NAME,
    resolve_mounted_lake,
)


def test_resolve_mounted_lake_standard_my_drive(tmp_path, monkeypatch):
    monkeypatch.delenv("DATA_LAKE_ROOT", raising=False)
    root = tmp_path / "MyDrive" / DEFAULT_LAKE_NAME
    root.mkdir(parents=True)
    assert resolve_mounted_lake(tmp_path) == root


def test_resolve_mounted_lake_explicit_override(tmp_path, monkeypatch):
    root = tmp_path / "MyDrive" / "shortcut-to-team-lake"
    root.mkdir(parents=True)
    monkeypatch.setenv("DATA_LAKE_ROOT", str(root))
    assert resolve_mounted_lake(tmp_path) == root
    another = tmp_path / "MyDrive" / DEFAULT_LAKE_NAME
    another.mkdir(parents=True)
    assert resolve_mounted_lake(tmp_path, override=another) == another


def test_missing_shared_drive_has_actionable_guidance(tmp_path, monkeypatch):
    monkeypatch.delenv("DATA_LAKE_ROOT", raising=False)
    with pytest.raises(FileNotFoundError, match="Cloning GitHub does not grant Drive"):
        resolve_mounted_lake(tmp_path)


def test_resolve_does_not_create_missing_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("DATA_LAKE_ROOT", raising=False)
    with pytest.raises(FileNotFoundError):
        resolve_mounted_lake(tmp_path, override=tmp_path / "nonexistent")
    assert not (tmp_path / "nonexistent").exists()
