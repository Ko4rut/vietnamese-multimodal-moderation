"""Offline tests for Gold V1: integrity, reproducibility and leakage mitigation."""

import hashlib
import json
from pathlib import Path

import pytest

from vietnamese_topic_classification.config import get_domain_labels
from vietnamese_topic_classification.data.base_reader import BaseReader
from vietnamese_topic_classification.data.gold import (
    GoldConfig,
    GoldIntegrityError,
    GoldLeakageError,
    GoldManifestError,
    GoldPipeline,
    GoldPublicationError,
    GoldReader,
    GoldSplitter,
    GoldValidationError,
)
from vietnamese_topic_classification.data.silver import SilverReader


def _raw_rows(n=48):
    labels = get_domain_labels()
    return [
        {
            "id": f"id-{i}",
            "text": f"Đây là bài viết riêng về chủ đề {labels[i % 3]} với mã số {i}. "
            f"Nội dung phục vụ dữ liệu văn bản tiếng Việt đầy đủ và dễ kiểm tra.",
            "domain": labels[i % 3],
            "text_length": 0,
            "source_snapshot": "bronze.jsonl",
            "quality_flags": [],
        }
        for i in range(n)
    ]


def _silver(tmp_path, rows=None):
    if rows is None:
        rows = _raw_rows()
    for row in rows:
        row["text_length"] = len(row["text"])
    silver_dir = tmp_path / "02_silver"
    silver_dir.mkdir(parents=True, exist_ok=True)
    data = silver_dir / "sample.jsonl"
    payload = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    ).encode("utf-8")
    data.write_bytes(payload)
    reports = tmp_path / "05_reports"
    reports.mkdir(parents=True, exist_ok=True)
    report = {
        "output_files": {"silver": data.name},
        "silver_sha256": hashlib.sha256(payload).hexdigest(),
        "accepted_rows": len(rows),
    }
    (reports / "silver_report.json").write_text(json.dumps(report), encoding="utf-8")
    return SilverReader(tmp_path, "silver_report.json")


def _reader_from_run(root: Path, run: dict):
    manifest_name = Path(run["paths"]["manifest"]).name
    return GoldReader(root, manifest_name)


def test_gold_pipeline_publishes_and_reads(tmp_path):
    input_reader = _silver(tmp_path)
    result = GoldPipeline(input_reader).run()
    reader = _reader_from_run(tmp_path, result)
    assert isinstance(reader, BaseReader)
    verification = reader.verify()
    assert verification["valid"]
    assert verification["records"] == 48
    assert all(verification["split_rows"][split] > 0 for split in ("train", "validation", "test"))
    assert result["manifest"]["label_to_id"] == {
        domain: i for i, domain in enumerate(get_domain_labels())
    }
    ids = {}
    for split in ("train", "validation", "test"):
        rows = list(reader.iter_split(split))
        assert len(rows) == verification["split_rows"][split]
        for row in rows:
            assert row["label_id"] == get_domain_labels().index(row["domain"])
            assert row["text_length"] == len(row["text"])
            assert row["id"] not in ids
            ids[row["id"]] = split
    assert len(ids) == 48
    assert len(list(reader.iter_records())) == 48
    assert result["report"]["near_duplicate_audit"]["exhaustive"] is False
    assert Path(result["paths"]["train"]).parent == tmp_path / "04_gold"
    assert Path(result["paths"]["validation"]).parent == tmp_path / "04_gold"
    assert Path(result["paths"]["test"]).parent == tmp_path / "04_gold"
    assert Path(result["paths"]["manifest"]).parent == tmp_path / "04_gold" / "manifests"
    assert Path(result["paths"]["report"]).parent == tmp_path / "05_reports"
    assert not (tmp_path / "03_gold").exists()
    assert set(result["paths"]) == {"train", "validation", "test", "manifest", "report"}


def test_gold_output_deterministic_across_roots(tmp_path):
    source = _silver(tmp_path)
    run_a = GoldPipeline(source, output_root=tmp_path / "one").run()
    run_b = GoldPipeline(source, output_root=tmp_path / "two").run()
    for split in ("train", "validation", "test"):
        a = Path(run_a["paths"][split]).read_bytes()
        b = Path(run_b["paths"][split]).read_bytes()
        assert a == b
    assert run_a["manifest"]["files"] == run_b["manifest"]["files"]


def test_refuses_overwrite(tmp_path):
    source = _silver(tmp_path)
    GoldPipeline(source).run()
    with pytest.raises(GoldPublicationError, match="already exist"):
        GoldPipeline(source).run()


def test_bad_silver_schema_does_not_publish(tmp_path):
    rows = _raw_rows(2)
    rows[1]["text"] = "This is a different text"
    rows[1]["text_length"] = -1
    source = _silver(tmp_path, rows)
    # Break after fixture writes valid lengths so Gold checks the contract.
    data = source.data_path
    bad = json.loads(data.read_text(encoding="utf-8").splitlines()[1])
    bad["text_length"] = -1
    payload = (
        json.dumps(rows[0], ensure_ascii=False) + "\n"
        + json.dumps(bad, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    data.write_bytes(payload)
    report = source.report_path
    metadata = json.loads(report.read_text(encoding="utf-8"))
    metadata["silver_sha256"] = hashlib.sha256(payload).hexdigest()
    report.write_text(json.dumps(metadata))
    source = SilverReader(tmp_path, "silver_report.json")
    with pytest.raises(GoldValidationError, match="text_length"):
        GoldPipeline(source).run()
    assert not (tmp_path / "04_gold").exists()
    assert not (tmp_path / "03_gold").exists()


def test_duplicate_silver_id_rejected(tmp_path):
    rows = _raw_rows(3)
    rows[1]["id"] = rows[0]["id"]
    with pytest.raises(GoldValidationError, match="duplicate Silver ID"):
        GoldPipeline(_silver(tmp_path, rows)).run()


def test_gold_reader_detects_tampering(tmp_path):
    run = GoldPipeline(_silver(tmp_path)).run()
    reader = _reader_from_run(tmp_path, run)
    split = reader.data_paths["train"]
    split.write_bytes(split.read_bytes() + b" \n")
    with pytest.raises(GoldIntegrityError, match="SHA256"):
        reader.verify()


def test_gold_reader_rejects_bad_manifest_or_path(tmp_path):
    with pytest.raises(GoldManifestError, match="filename"):
        GoldReader(tmp_path, "../bad.json")
    run = GoldPipeline(_silver(tmp_path)).run()
    metadata = Path(run["paths"]["manifest"])
    doc = json.loads(metadata.read_text(encoding="utf-8"))
    doc["files"]["train"]["file_name"] = "../escape.jsonl"
    metadata.write_text(json.dumps(doc))
    with pytest.raises(GoldManifestError, match="file entry"):
        GoldReader(tmp_path, metadata.name)


@pytest.mark.parametrize("config", [
    {"train_ratio": 0.9, "validation_ratio": 0.1, "test_ratio": 0.1},
    {"seed": -1},
    {"similarity_threshold": 0},
    {"shingle_words": 1},
    {"max_candidates_per_record": 0},
])
def test_invalid_gold_config(config):
    with pytest.raises(ValueError):
        GoldConfig(**config)


def test_near_duplicate_group_not_split():
    labels = get_domain_labels()
    base = " ".join(f"vietnamese{i}" for i in range(80))
    changed = base.replace("vietnamese20", "updatedtoken", 1)
    rows = [
        {"id": "a", "text": base, "domain": labels[0]},
        {"id": "b", "text": changed, "domain": labels[0]},
    ]
    rows.extend(
        {"id": f"independent{i}", "domain": labels[i % 3],
         "text": " ".join(f"unique{i}-{j}" for j in range(80))}
        for i in range(30)
    )
    config = GoldConfig(shingle_words=3, similarity_threshold=0.9, seed=13)
    allocations, audit = GoldSplitter(config).split(rows)
    a = next(split for split, indexes in allocations.items() if 0 in indexes)
    b = next(split for split, indexes in allocations.items() if 1 in indexes)
    assert a == b
    assert audit["matched_pairs"] >= 1
    assert not audit["exhaustive"]


def test_too_many_candidates_fails_closed():
    label = get_domain_labels()[0]
    base = " ".join(f"word{i}" for i in range(80))
    rows = [
        {"id": str(i), "text": base.replace("word20", f"replacement{i}"),
         "domain": label}
        for i in range(8)
    ]
    with pytest.raises(GoldLeakageError, match="candidate neighbors"):
        GoldSplitter(GoldConfig(shingle_words=3, max_candidates_per_record=1)).split(rows)


def test_empty_silver_cannot_create_gold(tmp_path):
    with pytest.raises(GoldValidationError, match="no accepted"):
        GoldPipeline(_silver(tmp_path, [])).run()


def test_gold_planned_paths_match_published_paths(tmp_path):
    """Colab preflight must use the same destination names as the CLI."""
    source = _silver(tmp_path)
    builder = GoldPipeline(source)
    expected = builder.planned_paths(source.verify()["sha256"])
    assert set(expected) == {"train", "validation", "test", "report", "manifest"}
    assert all(expected[split].parent == tmp_path / "04_gold" for split in ("train", "validation", "test"))
    assert expected["manifest"].parent == tmp_path / "04_gold" / "manifests"
    assert expected["report"].parent == tmp_path / "05_reports"
    assert not any(path.exists() for path in expected.values())
    result = builder.run()
    assert {name: str(path) for name, path in expected.items()} == result["paths"]
    assert all(path.is_file() for path in expected.values())
