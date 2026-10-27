"""Offline tests for Silver V1; no Drive or Hugging Face access."""

import hashlib
import json
from pathlib import Path

import pytest

from vietnamese_topic_classification.data.bronze import BronzeReader
from vietnamese_topic_classification.data.silver import (
    SilverPipeline,
    SilverPublicationError,
    SilverTransformer,
    SilverValidator,
)


def make_bronze(root: Path, records: list[dict]) -> BronzeReader:
    bronze = root / "01_bronze"
    bronze.mkdir()
    payload = b"".join(
        (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        for record in records
    )
    (bronze / "sample.jsonl").write_bytes(payload)
    metadata = root / "00_source_metadata"
    metadata.mkdir()
    (metadata / "snapshot.json").write_text(
        json.dumps({
            "file_name": "sample.jsonl",
            "sample_size": len(records),
            "file_size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }),
        encoding="utf-8",
    )
    return BronzeReader(root, "snapshot.json")


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def test_validator_rejects_bad_fields():
    validator = SilverValidator(domains=("News", "Games"))
    assert validator.validate({"id": "1", "text": "Hi", "domain": "News"}).valid
    assert validator.validate({"id": "", "text": "Hi", "domain": "News"}).reason == "invalid_id"
    assert validator.validate({"id": "1", "text": " ", "domain": "News"}).reason == "invalid_text"
    assert validator.validate({"id": "1", "text": "Hi", "domain": "Invalid"}).reason == "invalid_domain"


def test_transformer_normalizes_nfc_and_whitespace():
    assert SilverTransformer.normalize_text("  Ca\u0300\r\nXem\t tin  ") == "Cà\nXem tin"


def test_pipeline_deduplication_and_reporting(tmp_path):
    records = [
        {"id": "a", "text": " Ca\u0300\r\nXem\t tin ", "domain": "News"},
        {"id": "a", "text": "different", "domain": "News"},
        {"id": "b", "text": "Cà\nXem tin", "domain": "News"},
        {"id": "c", "text": "Cà\nXem tin", "domain": "Games"},
        {"id": "d", "text": "unique", "domain": "Games"},
        {"id": "e", "text": "", "domain": "News"},
        {"id": "f", "text": "hi", "domain": "other"},
    ]
    reader = make_bronze(tmp_path, records)
    result = SilverPipeline(reader, validator=SilverValidator(domains=("News", "Games"))).run()
    accepted = read_jsonl(result["paths"]["silver"])
    rejected = read_jsonl(result["paths"]["rejected"])
    assert [row["id"] for row in accepted] == ["a", "d"]
    assert accepted[0]["text"] == "Cà\nXem tin"
    assert accepted[0]["source_snapshot"] == "sample.jsonl"
    assert len(rejected) == 5
    assert result["report"]["input_rows"] == 7
    assert result["report"]["rejection_reasons"] == {
        "duplicate_id": 1, "duplicate_text": 1,
        "duplicate_text_conflicting_domain": 1,
        "invalid_domain": 1, "invalid_text": 1,
    }
    assert not list(tmp_path.rglob("*.uploading"))


def test_no_overwrite_and_no_bronze_modification(tmp_path):
    reader = make_bronze(tmp_path, [{"id": "1", "text": " abc ", "domain": "News"}])
    original = reader.data_path.read_bytes()
    pipeline = SilverPipeline(reader, validator=SilverValidator(domains=("News",)))
    pipeline.run()
    assert reader.data_path.read_bytes() == original
    with pytest.raises(SilverPublicationError):
        pipeline.run()


def test_verify_before_publishing(tmp_path):
    reader = make_bronze(tmp_path, [{"id": "1", "text": "abc", "domain": "News"}])
    reader.data_path.write_bytes(b'{"broken":true}\n')
    with pytest.raises(Exception, match="mismatch"):
        SilverPipeline(reader).run()
    assert not list(tmp_path.glob("02_silver/*.jsonl"))


def test_reject_all(tmp_path):
    reader = make_bronze(tmp_path, [{"id": "1", "text": " ", "domain": "News"}])
    result = SilverPipeline(reader).run()
    assert result["report"]["accepted_rows"] == 0
    assert result["report"]["text_length"] == {"min": None, "max": None, "mean": None}
    assert read_jsonl(result["paths"]["silver"]) == []
