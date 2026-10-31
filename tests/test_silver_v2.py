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

def test_structured_related_list_is_cleaned_without_copyright():
    body = "Bài viết giới thiệu chương trình học tại trường. " * 30
    footer = ("\nBài viết liên quan\n"
              "Cách chuẩn bị hồ sơ đại học\n"
              "Học phí theo từng chương trình\n"
              "Mẫu đơn xét tuyển mới nhất")
    result = SilverV2Cleaner().clean(body + footer)
    assert result.action == "cleaned"
    assert result.text == body.strip()
    assert "structured_related_list" in result.signals


def test_short_or_citation_related_sections_are_review_only():
    body = "Phân tích nghiên cứu khoa học. " * 30
    short = body + "\nBài viết liên quan\nCách đọc sách khoa học"
    citations = (body + "\nBài viết liên quan\nTài liệu tham khảo\n"
                 "Nghiên cứu A có kết quả\nNghiên cứu B có kết quả\n"
                 "Nghiên cứu C có kết quả")
    for text in (short, citations):
        result = SilverV2Cleaner().clean(text)
        assert result.action == "review"
        assert result.text == text


def test_footer_only_at_tail_and_guardrail_for_structured_links():
    body = "Nội dung chính về kinh tế trong nhiều năm. " * 23
    footer = ("\nTin liên quan\n" + "\n".join(
        "Bài viết về lãi suất ngân hàng trung ương" for _ in range(50)
    ))
    result = SilverV2Cleaner().clean(body + footer)
    assert result.action in ("unchanged", "guardrail")
    assert result.text == body + footer

def test_corpus_aware_repeated_ui_keeps_article_content():
    from vietnamese_topic_classification.data.silver.corpus_v2 import build_corpus_stats

    body = "Nội dung bài báo dài. " * 20
    source = "Skip to content\n" + body + "\nEmail *\n"
    corpus = build_corpus_stats([{"text": source}] * 4, min_document_frequency=2)
    result = SilverV2Cleaner().clean(source, corpus=corpus)
    assert "Skip to content" not in result.text
    assert "Email *" not in result.text
    assert "Nội dung bài báo dài." in result.text


def test_corpus_aware_preserves_references():
    from vietnamese_topic_classification.data.silver.corpus_v2 import build_corpus_stats

    source = "Tài liệu tham khảo\n" + "Nội dung chính. " * 30 + "\nKết luận"
    corpus = build_corpus_stats([{"text": source}] * 4, min_document_frequency=2)
    result = SilverV2Cleaner().clean(source, corpus=corpus)
    assert result.text == source


def test_v2_pipeline_corpus_prepass_removes_ui_from_published_text(tmp_path):
    body = "Bài báo về chủ đề văn hóa và lịch sử. " * 20
    reader = make_reader(tmp_path, [
        "Skip to content\n" + body + "\nEmail *"
        for _ in range(25)
    ])
    # Duplicate cleaned articles are correctly rejected after stripping.
    result = SilverV2Pipeline(reader).run(dry_run=True)
    assert result["report"]["input_rows"] == 25
    assert result["report"]["corpus_stats"]["documents"] == 25
    assert result["report"]["corpus_stats"]["repeated_ui_blocks"] >= 2
    assert result["report"]["cleaned_rows"] == 25
    assert result["report"]["accepted_rows"] == 1
    assert result["report"]["rejected_rows"] == 24


def test_silver_v2_output_paths_are_versioned(tmp_path):
    reader = make_reader(tmp_path, ["Đây là văn bản tin tức hợp lệ. " * 20])
    pipeline = SilverV2Pipeline(reader)
    paths = pipeline.planned_paths(reader.verify()["sha256"])
    assert paths["silver"].parent == tmp_path / "02_silver" / "v2"
    assert paths["rejected"].parent == tmp_path / "02_silver" / "v2"
    assert paths["audit"].parent == tmp_path / "05_reports" / "silver" / "v2"
    assert paths["report"].parent == tmp_path / "05_reports" / "silver" / "v2"


def test_silver_reader_supports_moved_legacy_v2_snapshot(tmp_path):
    text = "Văn bản được giữ nguyên. " * 20
    reader = make_reader(tmp_path, [text])
    source = reader.data_path
    report = json.loads(reader.report_path.read_text())
    report["pipeline_version"] = "v2"
    new_silver = tmp_path / "02_silver" / "v2"
    new_report = tmp_path / "05_reports" / "silver" / "v2"
    new_silver.mkdir(parents=True)
    new_report.mkdir(parents=True)
    source.rename(new_silver / source.name)
    reader.report_path.unlink()
    (new_report / "v1.json").write_text(json.dumps(report), encoding="utf-8")

    restored = SilverReader(tmp_path, "v1.json")
    assert restored.verify()["records"] == 1
    assert list(restored.iter_records())[0]["text"] == text


def test_silver_reader_handles_partial_v2_drive_migration(tmp_path):
    text = "Văn bản bài viết được bảo toàn. " * 20
    reader = make_reader(tmp_path, [text])
    report = json.loads(reader.report_path.read_text())
    report["pipeline_version"] = "v2"
    reader.report_path.write_text(json.dumps(report))
    data_dir = tmp_path / "02_silver" / "v2"
    data_dir.mkdir()
    reader.data_path.rename(data_dir / reader.data_path.name)

    assert SilverReader(tmp_path, "v1.json").verify()["records"] == 1


def test_silver_reader_rejects_ambiguous_quality_report(tmp_path):
    reader = make_reader(tmp_path, ["Một văn bản ngắn nhưng đúng cấu trúc. " * 8])
    nested = tmp_path / "05_reports" / "silver" / "v2"
    nested.mkdir(parents=True)
    (nested / "v1.json").write_bytes(reader.report_path.read_bytes())
    from vietnamese_topic_classification.data.silver.exceptions import SilverReportError

    with pytest.raises(SilverReportError, match="Ambiguous"):
        SilverReader(tmp_path, "v1.json")
