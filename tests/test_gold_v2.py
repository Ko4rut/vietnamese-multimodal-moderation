"""Test Gold V2 publication, source integrity, and version isolation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_gold import _silver

from vietnamese_topic_classification.config import get_domain_labels
from vietnamese_topic_classification.data.gold import (
    GoldReader,
    GoldV2Pipeline,
    GoldValidationError,
)
from vietnamese_topic_classification.data.silver import SilverReader
from vietnamese_topic_classification.data.silver.v2.pipeline import SilverV2Pipeline


def _published_source(tmp_path: Path) -> SilverReader:
    source = _silver(tmp_path)
    result = SilverV2Pipeline(source).run()
    reader = SilverReader(tmp_path, Path(result["paths"]["report"]).name)
    assert reader.verify()["records"] > 0
    return reader


def test_gold_v2_publish_verify_and_isolate(tmp_path):
    silver = _published_source(tmp_path)
    unchanged_v1 = (tmp_path / "02_silver" / "sample.jsonl").read_bytes()
    builder = GoldV2Pipeline(silver)
    verified = builder.verify_silver_publication()
    planned = builder.planned_paths(verified["sha256"])
    result = builder.run()
    assert result["paths"] == {key: str(path) for key, path in planned.items()}
    assert result["manifest"]["pipeline_version"] == "v2"
    assert result["manifest"]["source"]["silver_version"] == "v2"
    assert result["manifest"]["source"]["silver_v1_report"] == "silver_report.json"
    assert result["manifest"]["source"]["silver_report_sha256"]
    assert planned["manifest"].parent == tmp_path / "03_gold" / "v2" / "manifests"
    assert planned["report"].parent == tmp_path / "04_reports" / "gold" / "v2"
    assert not (tmp_path / "03_gold" / "manifests").exists()
    assert (tmp_path / "02_silver" / "sample.jsonl").read_bytes() == unchanged_v1
    gold = GoldReader(tmp_path, planned["manifest"].name)
    assert gold.version == "v2"
    counts = gold.verify()["split_rows"]
    assert sum(counts.values()) == result["manifest"]["total_rows"]
    assert all(counts[name] > 0 for name in ("train", "validation", "test"))
    ids = []
    labels = get_domain_labels()
    for name in ("train", "validation", "test"):
        for row in gold.iter_split(name):
            assert row["label_id"] == labels.index(row["domain"])
            assert row["text_length"] == len(row["text"])
            ids.append(row["id"])
    assert len(ids) == len(set(ids))
    with pytest.raises(Exception, match="already exist"):
        builder.run()


@pytest.mark.parametrize("missing", ["rejected", "audit"])
def test_gold_v2_rejects_partial_silver(tmp_path, missing):
    silver = _published_source(tmp_path)
    file_name = silver.report["output_files"][missing]
    folder = (
        tmp_path / "02_silver" / "v2"
        if missing == "rejected"
        else tmp_path / "04_reports" / "silver" / "v2"
    )
    (folder / file_name).unlink()
    with pytest.raises(GoldValidationError, match="Partial Silver V2"):
        GoldV2Pipeline(silver).run()
    assert not (tmp_path / "03_gold" / "v2").exists()


def test_gold_v2_rejects_changed_audit(tmp_path):
    silver = _published_source(tmp_path)
    audit = (
        tmp_path / "04_reports" / "silver" / "v2"
        / silver.report["output_files"]["audit"]
    )
    audit.write_bytes(audit.read_bytes() + b"\n")
    with pytest.raises(GoldValidationError, match="audit SHA256"):
        GoldV2Pipeline(silver).run()


def test_gold_v2_rejects_v1_source_and_tampered_lineage(tmp_path):
    source = _silver(tmp_path)
    with pytest.raises(GoldValidationError, match="requires a Silver V2"):
        GoldV2Pipeline(source)
    silver = _published_source(tmp_path)
    original = tmp_path / "02_silver" / "sample.jsonl"
    original.write_bytes(original.read_bytes() + b"\n")
    with pytest.raises(Exception, match="SHA256"):
        GoldV2Pipeline(silver).verify_silver_publication()


def test_gold_v2_reader_requires_matching_manifest_version(tmp_path):
    silver = _published_source(tmp_path)
    publication = GoldV2Pipeline(silver).run()
    manifest = Path(publication["paths"]["manifest"])
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    doc["pipeline_version"] = "v1"
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(Exception, match="pipeline_version"):
        GoldReader(tmp_path, manifest.name)
