"""Tests that must run without ML dependencies or network access."""

import pytest

from moderation.config import load_taxonomy
from moderation.inference.base import UnconfiguredModerator
from moderation.policy.engine import decide
from moderation.schemas import Action, Evidence, LabelScore, Modality, ModerationResult


def test_taxonomy_has_unique_sixteen_labels() -> None:
    taxonomy = load_taxonomy()
    ids = [label["id"] for label in taxonomy["labels"]]
    assert taxonomy["version"] == "mcst_v1"
    assert len(ids) == len(set(ids)) == 16


def test_evidence_interval_validation() -> None:
    with pytest.raises(ValueError):
        Evidence(source=Modality.VIDEO, description="clip", start_sec=4, end_sec=2)


def test_label_score_validation() -> None:
    with pytest.raises(ValueError):
        LabelScore(label_id="C01_HATE", score=1.2)


def test_missing_model_does_not_fake_predictions() -> None:
    with pytest.raises(NotImplementedError):
        UnconfiguredModerator().predict("demo", "example")


def test_policy_escalates_until_implemented() -> None:
    result = ModerationResult(
        content_id="sample",
        modality=Modality.TEXT,
        policy_version="mcst_v1",
    )
    assert result.recommended_action is None
    assert decide(result) == Action.ESCALATE
