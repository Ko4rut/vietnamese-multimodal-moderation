"""Offline integration and safety tests for Silver V2."""
import hashlib
import json

import pytest

from vietnamese_topic_classification.data.silver import SilverReader
from vietnamese_topic_classification.data.silver.cleaner_v2 import SilverV2Cleaner
from vietnamese_topic_classification.data.silver.exceptions import SilverPublicationError
from vietnamese_topic_classification.data.silver.pipeline_v2 import SilverV2Pipeline


def make_reader(tmp_path, texts):
    root = tmp_path / "02_silver"
    root.mkdir()
    rows = [
        {"id": str(i), "text": text, "domain": "News", "text_length": len(text),
         "quality_flags": [], "source_snapshot": "bronze.jsonl"}
        for i, text in enumerate(texts)
    ]
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    (root / "v1.jsonl").write_text(content, encoding="utf-8")
    (tmp_path / "05_reports").mkdir()
    (tmp_path / "05_reports" / "v1.json").write_text(json.dumps({
        "pipeline_version": "v1", "output_files": {"silver": "v1.jsonl"},
        "accepted_rows": len(rows),
        "silver_sha256": hashlib.sha256(content.encode()).hexdigest(),
    }))
    return SilverReader(tmp_path, "v1.json")


def test_cleaner_only_removes_guarded_terminal_suffix():
    body = "Nội dung chính về giáo dục và kinh tế. " * 25
    text = body + "\nBài viết liên quan\nMột vài tin khác\nCopyright © 2026"
    result = SilverV2Cleaner().clean(text)
    assert result.action == "cleaned"
    assert result.text == body.strip()
    assert result.removed_chars > 0
    assert "related_articles" in result.signals


def test_ambiguous_related_heading_and_middle_copyright_preserved():
    body = "Nội dung bài viết. " * 35
    assert SilverV2Cleaner().clean(body + "\nBài viết liên quan\nBài khác").action == "review"
    middle = body[:350] + "\nCopyright © 2026\n" + body[350:]
    assert SilverV2Cleaner().clean(middle).action == "unchanged"


def test_guardrail_prevents_large_content_loss():
    text = ("Bài chính. " * 80) + "\nCopyright © 2026\n" + ("Links " * 100)
    result = SilverV2Cleaner().clean(text)
    assert result.action == "guardrail"
    assert result.text == text


def test_v2_dry_run_publish_read_and_immutability(tmp_path):
    body = "Bài viết chính. " * 25
    reader = make_reader(tmp_path, [
        body + "\nCopyright © 2026",
        "Thông tin hợp lệ. " * 20,
        body + "\nCopyright © 2025",  # becomes a duplicate after cleaning
    ])
    v1_bytes = reader.data_path.read_bytes()
    pipe = SilverV2Pipeline(reader)
    planned = pipe.planned_paths(reader.verify()["sha256"])
    dry = pipe.run(dry_run=True)
    assert dry["dry_run"] is True
    assert dry["report"]["cleaned_rows"] == 2
    assert dry["report"]["rejected_rows"] == 1
    assert all(not path.exists() for path in planned.values())

    result = pipe.run()
    assert result["report"]["accepted_rows"] == 2
    assert result["report"]["input_rows"] == 3
    assert result["report"]["source_silver_sha256"] == reader.report["silver_sha256"]
    assert reader.data_path.read_bytes() == v1_bytes
    v2_reader = SilverReader(tmp_path, planned["report"].name)
    assert v2_reader.verify()["records"] == 2
    rows = list(v2_reader.iter_records())
    assert rows[0]["text"] == body.strip()
    assert "silver_v2:cleaned" in rows[0]["quality_flags"]
    audit = [json.loads(s) for s in planned["audit"].read_text().splitlines()]
    assert any(row["rejection_reason"] == "duplicate_text_after_cleaning" for row in audit)
    with pytest.raises(SilverPublicationError):
        pipe.run()


def test_v2_rejects_v2_as_source(tmp_path):
    reader = make_reader(tmp_path, ["A" * 250])
    data = json.loads(reader.report_path.read_text())
    data["pipeline_version"] = "v2"
    reader.report_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="requires a Silver V1"):
        SilverV2Pipeline(SilverReader(tmp_path, "v1.json"))
