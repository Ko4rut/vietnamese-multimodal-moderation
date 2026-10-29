"""Offline Gold V2 contract, source audit and immutable output tests."""

import hashlib
import json

import pytest

from vietnamese_topic_classification.config import get_domain_labels
from vietnamese_topic_classification.data.gold import (
    GoldReader,
    GoldValidationError,
)
from vietnamese_topic_classification.data.gold.exceptions import GoldPublicationError
from vietnamese_topic_classification.data.gold.pipeline_v2 import GoldV2Pipeline
from vietnamese_topic_classification.data.silver import SilverReader


def make_source(tmp_path, *, review=0, malformed=False):
    texts = [
        f"Văn bản chủ đề {i % 3}, nội dung riêng số {i}. "
        + "Đây là dữ liệu minh họa cho kiểm thử." for i in range(45)
    ]
    labels = get_domain_labels()
    data = tmp_path / "02_silver"
    data.mkdir()
    rows = []
    for i, text in enumerate(texts):
        rows.append({
            "id": f"row-{i}", "text": text, "text_length": len(text),
            "domain": labels[i % 3], "quality_flags": (
                ["silver_v2:review"] if review and i == 0 else []
            ), "source_snapshot": "source.jsonl",
        })
    bytes_ = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    (data / "silver_v2.jsonl").write_bytes(bytes_)
    reports = tmp_path / "05_reports"
    reports.mkdir()
    audit = b'{"action":"review"}\n' if review else b""
    (reports / "audit_v2.jsonl").write_bytes(audit)
    report = {
        "pipeline_version": "v2", "ruleset_version": "silver-v2-tail-v1",
        "output_files": {"silver": "silver_v2.jsonl", "audit": "audit_v2.jsonl"},
        "silver_sha256": hashlib.sha256(bytes_).hexdigest(),
        "audit_sha256": hashlib.sha256(audit).hexdigest(),
        "input_rows": 45, "accepted_rows": 45, "rejected_rows": 0,
        "cleaned_rows": 0, "review_rows": review, "guardrail_rows": 0,
    }
    if malformed:
        report["accepted_rows"] = 44
    (reports / "quality_v2.json").write_text(json.dumps(report))
    return SilverReader(tmp_path, "quality_v2.json")


def test_gold_v2_publish_integrity_and_versioning(tmp_path):
    source = make_source(tmp_path)
    pipeline = GoldV2Pipeline(source)
    planned = pipeline.planned_paths(source.verify()["sha256"])
    assert all("_v2_" in p.name for p in planned.values())
    result = pipeline.run()
    assert all(path.is_file() for path in planned.values())
    assert result["report"]["input_rows"] == 45
    reader = GoldReader(tmp_path, planned["manifest"].name)
    assert reader.verify()["records"] == 45
    assert reader.manifest["pipeline_version"] == "v2"
    assert result["manifest"]["source_silver_v2"]["ruleset_version"] == "silver-v2-tail-v1"
    ids = [row["id"] for split in ("train", "validation", "test")
           for row in reader.iter_split(split)]
    assert len(ids) == len(set(ids)) == 45
    with pytest.raises(GoldPublicationError, match="already exist"):
        pipeline.run()


def test_gold_v2_refuses_unreviewed_without_opt_in(tmp_path):
    source = make_source(tmp_path, review=1)
    with pytest.raises(GoldValidationError, match="review/guardrail"):
        GoldV2Pipeline(source).run()
    result = GoldV2Pipeline(source, allow_unreviewed=True).run()
    assert result["report"]["source_silver_v2"]["allow_unreviewed"] is True
    assert result["report"]["quality_flag_counts"]["silver_v2:review"] == 1


def test_gold_v2_refuses_unaudited_and_invalid_source(tmp_path):
    source = make_source(tmp_path)
    source.report["pipeline_version"] = "v1"
    with pytest.raises(GoldValidationError, match="Silver V2"):
        GoldV2Pipeline(source).run()
    source.report["pipeline_version"] = "v2"
    (tmp_path / "05_reports" / "audit_v2.jsonl").write_text("tamper")
    with pytest.raises(GoldValidationError, match="audit"):
        GoldV2Pipeline(source).run()


def test_gold_v2_bad_count_no_publication(tmp_path):
    source = make_source(tmp_path, malformed=True)
    with pytest.raises(GoldValidationError, match="counts"):
        GoldV2Pipeline(source).run()
    assert not (tmp_path / "04_gold").exists()
