"""Tests for the shared BaseReader contract and both reader entry points."""

import hashlib
import json

import pytest

from vietnamese_topic_classification.data.base_reader import BaseReader
from vietnamese_topic_classification.data.bronze import BronzeReader
from vietnamese_topic_classification.data.main import MANIFEST_NAME, REPORT_NAME, main, reader_test
from vietnamese_topic_classification.data.silver import SilverReader


def write_bronze(root, metadata_name=MANIFEST_NAME):
    (root / "01_bronze").mkdir(exist_ok=True)
    payload = b'{"id":"a","text":"hello","domain":"News"}\n'
    (root / "01_bronze" / "raw.jsonl").write_bytes(payload)
    (root / "00_source_metadata").mkdir(exist_ok=True)
    (root / "00_source_metadata" / metadata_name).write_text(json.dumps({
        "file_name": "raw.jsonl",
        "sample_size": 1,
        "file_size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }))
    return payload


def write_silver(root, metadata_name=REPORT_NAME):
    (root / "02_silver").mkdir(exist_ok=True)
    payload = b'{"id":"a","text":"hello","domain":"News","text_length":5}\n'
    (root / "02_silver" / "clean.jsonl").write_bytes(payload)
    (root / "04_reports").mkdir(exist_ok=True)
    (root / "04_reports" / metadata_name).write_text(json.dumps({
        "output_files": {"silver": "clean.jsonl"},
        "silver_sha256": hashlib.sha256(payload).hexdigest(),
        "accepted_rows": 1,
    }))
    return payload


def test_base_reader_is_abstract(tmp_path):
    with pytest.raises(TypeError):
        BaseReader(tmp_path, "sample.json")


@pytest.mark.parametrize("name", ["../a.json", "nested/a.json", "..\\a.json", "a.txt", ""])
def test_metadata_filename_validation(name):
    assert not BaseReader.is_safe_metadata_name(name)


def test_bronze_and_silver_share_constructor(tmp_path):
    write_bronze(tmp_path)
    write_silver(tmp_path)
    bronze = BronzeReader(data_lake_root=tmp_path, metadata_name=MANIFEST_NAME)
    silver = SilverReader(data_lake_root=tmp_path, metadata_name=REPORT_NAME)
    assert isinstance(bronze, BaseReader)
    assert isinstance(silver, BaseReader)
    assert bronze.root == silver.root == tmp_path
    assert bronze.metadata_name == MANIFEST_NAME
    assert silver.metadata_name == REPORT_NAME
    assert bronze.verify()["records"] == silver.verify()["records"] == 1


def test_generic_main_bronze(tmp_path, capsys):
    write_bronze(tmp_path)
    result = main(layer="bronze", data_lake_root=tmp_path)
    assert result["records"] == 1
    assert "BRONZE VALIDATION PASSED" in capsys.readouterr().out


def test_generic_main_silver(tmp_path, capsys):
    write_silver(tmp_path)
    result = main(layer="silver", data_lake_root=tmp_path)
    assert result["records"] == 1
    assert "SILVER VALIDATION PASSED" in capsys.readouterr().out


def test_generic_reader_handles_empty_silver(tmp_path, capsys):
    (tmp_path / "02_silver").mkdir()
    (tmp_path / "02_silver" / "clean.jsonl").write_bytes(b"")
    (tmp_path / "04_reports").mkdir()
    (tmp_path / "04_reports" / "quality.json").write_text(json.dumps({
        "output_files": {"silver": "clean.jsonl"},
        "silver_sha256": hashlib.sha256(b"").hexdigest(),
        "accepted_rows": 0,
    }))
    reader = SilverReader(tmp_path, "quality.json")
    assert reader_test(reader)["records"] == 0
    assert "First record" not in capsys.readouterr().out


def test_invalid_preview_limit_and_layer(tmp_path):
    write_bronze(tmp_path)
    reader = BronzeReader(tmp_path, MANIFEST_NAME)
    with pytest.raises(ValueError, match="nonnegative"):
        reader_test(reader, preview_limit=-1)
    with pytest.raises(ValueError, match="Unknown"):
        main(layer="gold", data_lake_root=tmp_path)
