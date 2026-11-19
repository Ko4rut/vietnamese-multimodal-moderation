from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_gold import _silver

from vietnamese_topic_classification.analysis.news_v3 import (
    NEWS_LABELS,
    build_keyword_report,
    build_prompt,
    load_v3_labels,
    sample_news_records,
    validate_decisions,
)
from vietnamese_topic_classification.data.gold import GoldReader, GoldV3Pipeline
from vietnamese_topic_classification.data.silver import SilverReader
from vietnamese_topic_classification.data.silver.v2.pipeline import SilverV2Pipeline
from vietnamese_topic_classification.data.silver.v3 import SilverV3Pipeline


def _published_silver_v2(tmp_path: Path) -> SilverReader:
    labels = [
        "News",
        "Sports",
        "Science",
        "News",
        "Law_and_Government",
        "Health",
        "News",
        "Finance",
        "Arts_and_Entertainment",
    ]
    rows = [
        {
            "id": f"row-{index}",
            "text": (
                f"Bài viết riêng số {index} về chủ đề {label}. "
                "Nội dung có thông tin và ngữ cảnh đầy đủ để làm dữ liệu thử nghiệm."
            ),
            "domain": label,
            "text_length": 0,
            "source_snapshot": "fixture.jsonl",
            "quality_flags": [],
        }
        for index, label in enumerate(labels)
    ]
    return SilverReader(
        tmp_path,
        Path(SilverV2Pipeline(_silver(tmp_path, rows)).run()["paths"]["report"]).name,
    )


def _decisions_for_news(source: SilverReader, path: Path, *, approved: bool = True) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in source.iter_records():
            if row["domain"] == "News":
                handle.write(json.dumps({
                    "id": row["id"],
                    "label": "News_Politics",
                    "needs_review": False,
                    "approved": approved,
                    "reviewer": "reviewer-1",
                    "reason": "Fixture decision",
                }, ensure_ascii=False) + "\n")
    return path


def test_v3_taxonomy_is_candidate_split_of_news() -> None:
    labels = load_v3_labels()
    assert "News" not in labels
    assert len(labels) == 27
    assert "News_Sports" not in labels
    assert len(NEWS_LABELS) == 3
    assert set(NEWS_LABELS).issubset(labels)


def test_keyword_report_surfaces_class_terms_and_overlap_without_relabeling() -> None:
    report = build_keyword_report([
        {"id": "n1", "domain": "News", "text": "Bầu cử quốc hội và chính sách giáo dục."},
        {"id": "n2", "domain": "News", "text": "Quốc hội thảo luận chính sách y tế."},
        {"id": "s1", "domain": "Sports", "text": "Quốc hội thảo luận giải đấu bóng đá."},
    ], top_n=5)

    assert report["keyword_based_label_assignment"] is False
    assert report["classes"]["News"]["documents"] == 2
    assert report["classes"]["News"]["examples"][0]["id"] == "n1"
    assert "Sports" in report["news_vs_other_class_keyword_overlap"]
    assert report["classes"]["News"]["keywords"]


def test_news_pilot_is_seeded_and_does_not_include_other_domains() -> None:
    rows = [
        {"id": f"news-{index}", "domain": "News", "text": f"News article {index}"}
        for index in range(10)
    ] + [{"id": "sports-1", "domain": "Sports", "text": "Sports article"}]

    sample = sample_news_records(rows, sample_size=4, seed=7)
    assert sample == sample_news_records(rows, sample_size=4, seed=7)
    assert len(sample) == 4
    assert all(row["domain"] == "News" for row in sample)
    assert len({row["id"] for row in sample}) == 4


def test_prompt_is_strict_and_machine_readable() -> None:
    prompt = build_prompt({"text": "Tổng thống ký sắc lệnh"})
    assert "Chỉ chọn đúng một nhãn" in prompt
    assert '"label"' in prompt
    assert "News_Politics" in prompt
    assert "News_Sports" not in prompt
    assert "Sports" in prompt
    truncated = build_prompt({"text": "x" * 20}, max_chars=10)
    assert "[Đã lược bớt phần còn lại của bài viết.]" in truncated
    with pytest.raises(ValueError, match="max_chars must be positive"):
        build_prompt({"text": "article"}, max_chars=0)


def test_validate_decisions_rejects_unknown_and_duplicate() -> None:
    rows = [{"id": "1", "label": "News_Politics", "confidence": 0.9}]
    assert validate_decisions(rows)[0]["needs_review"] is False
    with pytest.raises(ValueError, match="Unknown V3 label"):
        validate_decisions([{"id": "1", "label": "News", "confidence": 0.9}])
    with pytest.raises(ValueError, match="Unknown V3 label"):
        validate_decisions([{"id": "2", "label": "News_Sports", "confidence": 0.9}])
    with pytest.raises(ValueError, match="Duplicate decision"):
        validate_decisions([
            {"id": "1", "label": "News_Politics", "confidence": 0.9},
            {"id": "1", "label": "News_Sports", "confidence": 0.8},
        ])


def test_silver_v3_relabels_then_gold_v3_splits_from_silver(tmp_path: Path) -> None:
    source = _published_silver_v2(tmp_path)
    decisions = _decisions_for_news(source, tmp_path / "decisions.jsonl")
    silver_result = SilverV3Pipeline(source, decisions).run()
    silver_v3 = SilverReader(tmp_path, Path(silver_result["paths"]["report"]).name)
    assert SilverV3Pipeline.verify_reader(silver_v3)["records"] == len(list(source.iter_records()))
    assert silver_v3.report["source"]["silver_v2_report"] == source.report_path.name
    assert silver_v3.report["relabeled_rows"] == 3
    assert all(row["domain"] != "News" for row in silver_v3.iter_records())

    gold_result = GoldV3Pipeline(silver_v3).run()
    gold_v3 = GoldReader(tmp_path, Path(gold_result["paths"]["manifest"]).name)
    verified = gold_v3.verify()
    assert gold_v3.version == "v3"
    assert verified["records"] == silver_v3.report["accepted_rows"]
    assert all(count > 0 for count in verified["split_rows"].values())
    assert gold_v3.manifest["source"]["silver_version"] == "v3"
    assert gold_v3.manifest["source"]["silver_v2_report"] == source.report_path.name
    assert Path(gold_result["paths"]["train"]).parent == tmp_path / "03_gold" / "v3"


def test_silver_v3_rejects_unapproved_news_and_does_not_publish(tmp_path: Path) -> None:
    source = _published_silver_v2(tmp_path)
    decisions = _decisions_for_news(
        source, tmp_path / "unapproved.jsonl", approved=False,
    )
    with pytest.raises(ValueError, match="not human-approved"):
        SilverV3Pipeline(source, decisions).run()
    assert not (tmp_path / "02_silver" / "v3").exists()
