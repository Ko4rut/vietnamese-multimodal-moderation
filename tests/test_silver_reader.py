"""Offline tests for SilverReader, without a Drive mount or network access."""

import hashlib
import json
from pathlib import Path

import pytest

from vietnamese_topic_classification.data.base_reader import BaseReader
from vietnamese_topic_classification.data.silver import SilverReader
from vietnamese_topic_classification.data.silver.exceptions import (
    SilverFileNotFoundError,
    SilverFormatError,
    SilverIntegrityError,
    SilverReportError,
)


def make_silver(root, records=None, payload=None):
    if records is None:
        records = [{"id": "a", "text": "Xin chào", "domain": "News", "text_length": 8}]
    if payload is None:
        payload = b"".join(
            (json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8") for item in records
        )
    silver = root / "02_silver"
    silver.mkdir()
    (silver / "silver.jsonl").write_bytes(payload)
    reports = root / "04_reports"
    reports.mkdir()
    report = {
        "output_files": {"silver": "silver.jsonl", "rejected": "rejected.jsonl"},
        "silver_sha256": hashlib.sha256(payload).hexdigest(),
        "accepted_rows": len(records),
    }
    (reports / "quality.json").write_text(json.dumps(report), encoding="utf-8")
    return report


def test_verify_and_stream(tmp_path):
    report = make_silver(tmp_path)
    reader = SilverReader(tmp_path, "quality.json")
    assert isinstance(reader, BaseReader)
    assert reader.verify() == {
        "valid": True,
        "records": 1,
        "sha256": report["silver_sha256"],
    }
    assert list(reader.iter_records())[0]["text"] == "Xin chào"


def test_empty_accepted_snapshot(tmp_path):
    make_silver(tmp_path, records=[])
    reader = SilverReader(tmp_path, "quality.json")
    assert reader.verify()["records"] == 0
    assert list(reader.iter_records()) == []


def test_missing_and_bad_report(tmp_path):
    with pytest.raises(SilverReportError, match="quality report"):
        SilverReader(tmp_path, "missing.json")
    make_silver(tmp_path)
    (tmp_path / "04_reports" / "quality.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(SilverReportError, match="JSON object"):
        SilverReader(tmp_path, "quality.json")


@pytest.mark.parametrize("bad_name", ["../quality.json", "sub/quality.json", "..\\quality.json", "quality.txt", ""])
def test_report_name_rejects_paths(tmp_path, bad_name):
    with pytest.raises(SilverReportError, match="filename"):
        SilverReader(tmp_path, bad_name)


@pytest.mark.parametrize("bad_filename", ["../silver.jsonl", "sub/silver.jsonl", "..\\silver.jsonl", "foo.txt", None])
def test_report_rejects_unsafe_filename(tmp_path, bad_filename):
    report = make_silver(tmp_path)
    report["output_files"]["silver"] = bad_filename
    (tmp_path / "04_reports" / "quality.json").write_text(json.dumps(report))
    with pytest.raises(SilverReportError, match="filename"):
        SilverReader(tmp_path, "quality.json")


def test_report_field_validation(tmp_path):
    report = make_silver(tmp_path)
    for change in ({"accepted_rows": -1}, {"accepted_rows": True}, {"silver_sha256": "bad"}, {"output_files": []}):
        changed = {**report, **change}
        (tmp_path / "04_reports" / "quality.json").write_text(json.dumps(changed))
        with pytest.raises(SilverReportError):
            SilverReader(tmp_path, "quality.json")


def test_missing_silver_file(tmp_path):
    make_silver(tmp_path)
    (tmp_path / "02_silver" / "silver.jsonl").unlink()
    with pytest.raises(SilverFileNotFoundError):
        SilverReader(tmp_path, "quality.json")


def test_checksum_mismatch(tmp_path):
    make_silver(tmp_path)
    path = tmp_path / "02_silver" / "silver.jsonl"
    path.write_bytes(path.read_bytes().replace(b"News", b"Sport"))
    with pytest.raises(SilverIntegrityError, match="SHA256 mismatch"):
        SilverReader(tmp_path, "quality.json").verify()


def test_count_mismatch(tmp_path):
    report = make_silver(tmp_path)
    report["accepted_rows"] = 2
    (tmp_path / "04_reports" / "quality.json").write_text(json.dumps(report))
    with pytest.raises(SilverIntegrityError, match="record count mismatch"):
        SilverReader(tmp_path, "quality.json").verify()


@pytest.mark.parametrize("payload,match", [(b'{"bad":\n', "line 1"), (b'[1,2]\n', "JSON object"), (b'\n', "line 1")])
def test_malformed_jsonl(tmp_path, payload, match):
    make_silver(tmp_path, payload=payload)
    with pytest.raises(SilverFormatError, match=match):
        list(SilverReader(tmp_path, "quality.json").iter_records())


def test_file_removed_after_initialization(tmp_path):
    make_silver(tmp_path)
    reader = SilverReader(tmp_path, "quality.json")
    reader.data_path.unlink()
    with pytest.raises(SilverFileNotFoundError):
        list(reader.iter_records())


def test_real_silver_pipeline_output_contract(tmp_path):
    """Build a Silver snapshot, then read its published quality report."""
    from vietnamese_topic_classification.data.bronze import BronzeReader
    from vietnamese_topic_classification.data.silver import SilverPipeline

    bronze = tmp_path / "01_bronze"
    bronze.mkdir()
    payload = b'{"id":"a","text":"Xin chao","domain":"News"}\n'
    (bronze / "sample.jsonl").write_bytes(payload)
    source_meta = tmp_path / "00_source_metadata"
    source_meta.mkdir()
    (source_meta / "snapshot.json").write_text(json.dumps({
        "file_name": "sample.jsonl",
        "sample_size": 1,
        "file_size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }))
    published = SilverPipeline(BronzeReader(tmp_path, "snapshot.json")).run()
    reader = SilverReader(tmp_path, Path(published["paths"]["report"]).name)
    assert reader.verify()["records"] == published["report"]["accepted_rows"] == 1
    assert [item["id"] for item in reader.iter_records()] == ["a"]
