"""Verify Silver version separation and backward-compatible snapshot readers."""

from __future__ import annotations

import hashlib
import json

import pytest

from vietnamese_topic_classification.data.silver import SilverPipeline, SilverReader
from vietnamese_topic_classification.data.silver.exceptions import SilverReportError
from vietnamese_topic_classification.data.silver.v1 import SilverPipeline as V1Pipeline
from vietnamese_topic_classification.data.silver.v2 import SilverV2Cleaner, SilverV2Pipeline


def _write_versioned(root, version: str, *, legacy: bool = False):
    payload = b'{"id":"1","text":"hello","domain":"News"}\n'
    data_dir = root / "02_silver" / version
    report_dir = root / "05_reports" / "silver" / version
    if legacy:
        data_dir = root / "02_silver"
        report_dir = root / "05_reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    data_name = f"data_{version}.jsonl"
    report_name = f"quality_{version}.json"
    (data_dir / data_name).write_bytes(payload)
    report = {
        "pipeline_version": version,
        "output_files": {"silver": data_name},
        "silver_sha256": hashlib.sha256(payload).hexdigest(),
        "accepted_rows": 1,
    }
    (report_dir / report_name).write_text(json.dumps(report), encoding="utf-8")
    return report_name


def test_canonical_versioned_imports():
    assert SilverPipeline is V1Pipeline
    assert SilverV2Cleaner.__module__.endswith('.v2.cleaner')
    assert SilverV2Pipeline.__module__.endswith('.v2.pipeline')


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("legacy", [True, False])
def test_reader_loads_legacy_and_versioned_layout(tmp_path, version, legacy):
    name = _write_versioned(tmp_path, version, legacy=legacy)
    reader = SilverReader(tmp_path, name)
    assert reader.verify()["records"] == 1
    assert reader.report["pipeline_version"] == version
    assert list(reader.iter_records())[0]["id"] == "1"


def test_reader_refuses_ambiguous_snapshot_locations(tmp_path):
    name = _write_versioned(tmp_path, "v1")
    legacy = tmp_path / "05_reports" / name
    legacy.write_bytes((tmp_path / "05_reports" / "silver" / "v1" / name).read_bytes())
    with pytest.raises(SilverReportError, match="Ambiguous"):
        SilverReader(tmp_path, name)


def test_reader_refuses_conflicting_data_duplicates(tmp_path):
    name = _write_versioned(tmp_path, "v2")
    source = tmp_path / "02_silver" / "v2" / "data_v2.jsonl"
    (tmp_path / "02_silver" / source.name).write_bytes(source.read_bytes())
    with pytest.raises(SilverReportError, match="Ambiguous"):
        SilverReader(tmp_path, name)
