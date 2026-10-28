"""Layer-specific models describe existing JSON contracts without changing payloads."""

import json
from dataclasses import is_dataclass
from typing import get_type_hints

from vietnamese_topic_classification.data.bronze import BronzeManifest, BronzeRecord
from vietnamese_topic_classification.data.bronze.model import (
    BronzeIngestResult,
    BronzeIngestionReport,
    BronzePublishedManifest,
)
from vietnamese_topic_classification.data.gold import GoldManifest, GoldRecord
from vietnamese_topic_classification.data.gold.model import GoldPipelineResult, GoldQualityReport
from vietnamese_topic_classification.data.silver import SilverQualityReport, SilverRecord
from vietnamese_topic_classification.data.silver.model import (
    SilverPublishedReport,
    SilverRejectedRecord,
    SilverRunResult,
    ValidationResult,
)
from vietnamese_topic_classification.data.silver.validator import (
    ValidationResult as LegacyValidationResult,
)


def test_records_match_existing_json_contracts():
    raw = BronzeRecord(id="a", text=None, domain="Games")
    clean = SilverRecord(
        id="a",
        text="Xin chào",
        domain="Games",
        text_length=8,
        source_snapshot="raw.jsonl",
        quality_flags=[],
    )
    gold = GoldRecord(**clean, label_id=8)
    assert json.loads(json.dumps(raw)) == raw
    assert json.loads(json.dumps(clean, ensure_ascii=False)) == clean
    assert json.loads(json.dumps(gold, ensure_ascii=False)) == gold
    assert set(gold) == set(clean) | {"label_id"}
    assert get_type_hints(GoldRecord)["label_id"] is int


def test_models_expose_snapshot_contracts():
    assert {"file_name", "sha256", "sample_size", "file_size_bytes"} <= (
        BronzeManifest.__required_keys__
    )
    assert {"output_files", "silver_sha256", "accepted_rows"} <= (
        SilverQualityReport.__required_keys__
    )
    assert "total_rows" in GoldManifest.__required_keys__
    assert "files" in GoldManifest.__required_keys__
    assert "source" in GoldQualityReport.__required_keys__
    assert "report" in GoldPipelineResult.__required_keys__
    assert "report" in SilverRunResult.__required_keys__
    assert "count" in BronzeIngestResult.__required_keys__
    assert "domain_counts" in BronzeIngestionReport.__required_keys__
    assert "dataset" not in BronzePublishedManifest.__required_keys__
    assert "source_manifest" not in SilverPublishedReport.__required_keys__
    assert "source_line" in SilverRejectedRecord.__required_keys__


def test_validation_result_backward_compatible():
    assert ValidationResult is LegacyValidationResult
    assert is_dataclass(ValidationResult)
    assert ValidationResult(True).reason is None
    assert ValidationResult(False, "invalid_text").valid is False
