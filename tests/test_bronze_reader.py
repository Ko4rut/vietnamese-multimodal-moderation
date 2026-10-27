"""Offline BronzeReader tests: no Colab mount or Drive API required."""

import hashlib
import json

import pytest

from vietnamese_topic_classification.data.bronze_reader import BronzeReader
from vietnamese_topic_classification.data.exceptions.bronze_exception import (
    BronzeException,
    BronzeFileNotFoundError,
    BronzeFormatError,
    BronzeIntegrityError,
    BronzeManifestError,
)


def write_snapshot(root, payload=b'{"id":"a","text":"Xin chao","domain":"News"}\n'):
    bronze = root / "01_bronze"
    bronze.mkdir()
    path = bronze / "sample.jsonl"
    path.write_bytes(payload)
    meta = root / "00_source_metadata"
    meta.mkdir()
    manifest = {
        "file_name": path.name,
        "sample_size": payload.count(b"\n"),
        "file_size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    (meta / "snapshot.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def test_verify_and_iterate(tmp_path):
    write_snapshot(tmp_path)
    reader = BronzeReader(tmp_path, "snapshot.json")
    assert reader.verify()["records"] == 1
    assert list(reader.iter_records()) == [{"id": "a", "text": "Xin chao", "domain": "News"}]


def test_missing_manifest(tmp_path):
    with pytest.raises(BronzeManifestError):
        BronzeReader(tmp_path, "snapshot.json")


def test_bad_manifest(tmp_path):
    path, manifest = write_snapshot(tmp_path)
    manifest["file_name"] = "../bad.jsonl"
    (tmp_path / "00_source_metadata" / "snapshot.json").write_text(json.dumps(manifest))
    with pytest.raises(BronzeManifestError):
        BronzeReader(tmp_path, "snapshot.json")


def test_missing_bronze(tmp_path):
    path, _ = write_snapshot(tmp_path)
    path.unlink()
    with pytest.raises(BronzeFileNotFoundError):
        BronzeReader(tmp_path, "snapshot.json")


def test_checksum_mismatch(tmp_path):
    path, manifest = write_snapshot(tmp_path)
    path.write_bytes(path.read_bytes().replace(b"Xin", b"Tin"))
    with pytest.raises(BronzeIntegrityError, match="SHA256"):
        BronzeReader(tmp_path, "snapshot.json").verify()


def test_size_mismatch(tmp_path):
    path, _ = write_snapshot(tmp_path)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(BronzeIntegrityError, match="size"):
        BronzeReader(tmp_path, "snapshot.json").verify()


def test_record_count_mismatch(tmp_path):
    path, manifest = write_snapshot(tmp_path)
    manifest["sample_size"] = 2
    (tmp_path / "00_source_metadata" / "snapshot.json").write_text(json.dumps(manifest))
    with pytest.raises(BronzeIntegrityError, match="count"):
        BronzeReader(tmp_path, "snapshot.json").verify()


def test_malformed_jsonl(tmp_path):
    write_snapshot(tmp_path, b'{"id":\n')
    with pytest.raises(BronzeFormatError, match="line 1"):
        list(BronzeReader(tmp_path, "snapshot.json").iter_records())


def test_wrong_json_type(tmp_path):
    write_snapshot(tmp_path, b'[1, 2]\n')
    with pytest.raises(BronzeFormatError, match="object"):
        list(BronzeReader(tmp_path, "snapshot.json").iter_records())


def test_bronze_errors_share_common_base():
    assert issubclass(BronzeManifestError, BronzeException)
    assert issubclass(BronzeFileNotFoundError, BronzeException)
    assert issubclass(BronzeIntegrityError, BronzeException)
    assert issubclass(BronzeFormatError, BronzeException)
