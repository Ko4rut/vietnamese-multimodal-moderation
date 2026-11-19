from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "15_label_news_v3_local.py"
SPEC = importlib.util.spec_from_file_location("local_news_labeler_v3", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_local_model_response_requires_valid_v3_label_and_confidence() -> None:
    labels = {"Sports", "News_Politics"}
    parsed = MODULE._parse_response(
        '```json\n{"label":"Sports","confidence":0.91,'
        '"needs_review":false,"reason":"Sports is the primary topic."}\n```',
        labels,
    )
    assert parsed["label"] == "Sports"
    assert parsed["confidence"] == pytest.approx(0.91)

    with pytest.raises(ValueError, match="unknown V3 label"):
        MODULE._parse_response(
            '{"label":"News_Sports","confidence":0.9,'
            '"needs_review":false,"reason":"sports"}',
            labels,
        )
    with pytest.raises(ValueError, match="invalid confidence"):
        MODULE._parse_response(
            '{"label":"Sports","confidence":true,'
            '"needs_review":false,"reason":"sports"}',
            labels,
        )


def test_local_model_candidate_is_never_auto_approved() -> None:
    candidate = MODULE._candidate_row(
        {"id": "news-1", "text": "full article text"},
        '{"label":"Sports","confidence":0.91,'
        '"needs_review":false,"reason":"Sports is the primary topic."}',
        allowed_labels={"Sports"},
        model_name="local-model",
        model_revision="revision-1",
        input_truncated=True,
    )
    assert candidate["status"] == "candidate"
    assert candidate["label"] == "Sports"
    assert candidate["approved"] is False
    assert candidate["needs_review"] is True
    assert candidate["input_truncated"] is True
    assert "inspect the full article" in candidate["reason"]


def test_resume_refuses_reviewed_or_non_candidate_files(tmp_path: Path) -> None:
    path = tmp_path / "decisions.jsonl"
    path.write_text(
        json.dumps({
            "id": "news-1",
            "status": "candidate",
            "label": "Sports",
            "confidence": 0.9,
            "approved": True,
            "needs_review": False,
            "reviewer": "reviewer-1",
        }) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="human-reviewed"):
        MODULE._load_existing(path, {"Sports"})

    path.write_text(json.dumps({"id": "news-1", "label": ""}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a local-model candidate file"):
        MODULE._load_existing(path, {"Sports"})
