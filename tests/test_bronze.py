"""Bronze offline tests: never access Hugging Face or Google Drive."""

import hashlib
import json

import pytest

from vietnamese_topic_classification.data.bronze import BronzeConfig, BronzeIngestor


def test_ingest_preserves_source_records_and_publishes_manifest(tmp_path, monkeypatch):
    rows = [
        {"id": "01", "text": "  Tiếng Việt  ", "domain": "Science", "extra": "ignored"},
        {"id": "02", "text": "Xin chào\nViệt Nam", "domain": "News"},
        {"id": "03", "text": "", "domain": "Science"},
    ]
    monkeypatch.setattr(BronzeIngestor, "_source_rows", lambda self: iter(rows))
    config = BronzeConfig(output_root=tmp_path, sample_size=3, shuffle_buffer=10)
    result = BronzeIngestor(config).run()

    bronze = tmp_path / "01_bronze"
    [raw] = bronze.glob("*.jsonl")
    parsed = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()]
    assert parsed == [
        {"id": "01", "text": "  Tiếng Việt  ", "domain": "Science"},
        {"id": "02", "text": "Xin chào\nViệt Nam", "domain": "News"},
        {"id": "03", "text": "", "domain": "Science"},
    ]
    [manifest_file] = (tmp_path / "00_source_metadata").glob("*.json")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["revision"] == "b81fcce58945970117a1b56d50ec81be2628a5c3"
    assert manifest["file_size_bytes"] == raw.stat().st_size
    assert manifest["sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert result["domain_counts"] == {"News": 1, "Science": 2}
    assert result["count"] == 3
    assert sorted(p.name for p in tmp_path.rglob("*.uploading")) == []


def test_missing_fields_stay_in_bronze_to_be_checked_by_silver(tmp_path, monkeypatch):
    monkeypatch.setattr(
        BronzeIngestor, "_source_rows", lambda self: iter([{"id": "a", "text": "abc"}])
    )
    result = BronzeIngestor(BronzeConfig(output_root=tmp_path, sample_size=1)).run()
    assert result["missing_field_counts"] == {"domain": 1}
    assert result["domain_counts"] == {"<missing>": 1}


def test_existing_snapshot_is_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(
        BronzeIngestor, "_source_rows", lambda self: iter([{"id": "a", "text": "a", "domain": "News"}])
    )
    ingestor = BronzeIngestor(BronzeConfig(output_root=tmp_path, sample_size=1))
    ingestor.run()
    with pytest.raises(FileExistsError):
        ingestor.run()


def test_short_stream_does_not_publish_partial_data(tmp_path, monkeypatch):
    monkeypatch.setattr(BronzeIngestor, "_source_rows", lambda self: iter([{"id": "a"}]))
    with pytest.raises(ValueError, match="Source ended"):
        BronzeIngestor(BronzeConfig(output_root=tmp_path, sample_size=2)).run()
    assert not list(tmp_path.rglob("*.jsonl"))
    assert not list(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("size,buffer", [(0, 100), (10, 1)])
def test_invalid_config_rejected(tmp_path, size, buffer):
    with pytest.raises(ValueError):
        BronzeConfig(output_root=tmp_path, sample_size=size, shuffle_buffer=buffer)
