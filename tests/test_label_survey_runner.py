"""End-to-end offline run of the survey on fake shards, plus Data Lake inputs."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("sklearn")

from conftest import FAKE_CANONICAL  # noqa: E402

from vietnamese_topic_classification.analysis.label_survey import lake_inputs, report  # noqa: E402
from vietnamese_topic_classification.analysis.label_survey.config import SurveyConfig  # noqa: E402
from vietnamese_topic_classification.analysis.label_survey.hf_parquet import (  # noqa: E402
    LocalParquetSource,
)
from vietnamese_topic_classification.analysis.label_survey.runner import SurveyRun  # noqa: E402

EXPECTED_FILES = [
    "dataset_metadata.csv", "full_label_counts.csv", "file_position_label_pct.csv",
    "label_position_drift.csv", "survey_sample_label_counts.csv", "survey_sample_vs_full.csv",
    "length_profile.csv", "conflicting_duplicates.csv", "top_terms_by_label.csv",
    "confusion_matrix_oof.csv", "confusing_pairs.csv", "per_label_metrics_oof.csv",
    "head_vs_full.csv", "suspected_mislabels_by_label.csv", "label_scenarios.csv",
    "audit_pack.csv", "audit_instructions.md", "summary.json", "label_survey_report.md",
    "fig_full_label_distribution.png", "fig_label_by_file_position.png",
    "fig_length_by_label.png", "fig_confusion_matrix_oof.png", "fig_top_confusing_pairs.png",
    "fig_label_dendrogram.png", "fig_per_label_f1.png", "fig_greedy_merge_curve.png",
]


def tiny_config(**overrides) -> SurveyConfig:
    values = dict(survey_target_rows=1200, per_class_cap=150, n_folds=3, top_pairs=3,
                  lr_max_iter=100, http_workers=4, n_jobs=1, max_features=2000,
                  audit_per_class=3, audit_per_pair=4, audit_part_c=5, head_chars=200,
                  text_char_limit=500)
    values.update(overrides)
    return SurveyConfig(**values)


def make_run(shards: Path, root: Path, **kwargs) -> SurveyRun:
    return SurveyRun(tiny_config(), root, source=LocalParquetSource(shards),
                     canonical_labels=FAKE_CANONICAL, log=lambda *_: None, **kwargs)


@pytest.fixture()
def finished(fake_shards, tmp_path):
    run = make_run(fake_shards["directory"], tmp_path / "out")
    zip_path = run.run_all()
    return run, zip_path


def test_end_to_end_outputs_and_zip(finished) -> None:
    run, zip_path = finished
    for name in EXPECTED_FILES:
        assert (run.out_dir / name).is_file(), name
    assert report.is_complete(run.out_dir)
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
    assert all(n.startswith(run.run_id + "/") for n in names)
    assert f"{run.run_id}/audit_pack.csv" in names and f"{run.run_id}/summary.json" in names
    summary = json.loads((run.out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["full_dataset"]["total_rows"] == 1806
    assert summary["full_dataset"]["adult_count"] == 1 and summary["full_dataset"]["null_count"] == 3
    assert summary["bronze_vs_full"]["status"] == "skipped"
    assert summary["survey_sample"]["strategy"]["method"].startswith("random_row_groups")
    assert summary["config"]["seed"] == 42 and summary["environment"]["libraries"]["pandas"]
    assert set(summary["timings_seconds"]) >= {"full_counts", "sample", "model", "audit"}
    text = (run.out_dir / "label_survey_report.md").read_text(encoding="utf-8")
    assert "bỏ qua" in text  # a missing Bronze comparison is stated, never silently equal
    # the only "nên gộp" is the mandatory disclaimer "điểm tăng không có nghĩa là nên gộp"
    assert text.count("nên gộp") == 1 and "có nghĩa là nên gộp" in text
    assert "nên bỏ" not in text and "nên giữ" not in text


def test_outputs_do_not_leak_article_text(finished) -> None:
    run, _zip = finished
    for name in ("summary.json", "label_survey_report.md"):
        body = (run.out_dir / name).read_text(encoding="utf-8")
        assert "Bài trùng nội dung" not in body and "chung12" not in body
    conflicts = pd.read_csv(run.out_dir / "conflicting_duplicates.csv")
    assert list(conflicts.columns) == ["group", "id", "domain"] and len(conflicts) > 0
    pack = pd.read_csv(run.out_dir / "audit_pack.csv", encoding="utf-8-sig")
    assert pack["excerpt"].str.len().max() <= 601


def test_label_distribution_counts_unexpected_values(finished) -> None:
    run, _ = finished
    table = run.table("full_label_counts").set_index("label")
    assert table.loc["Adult", "kind"] == "unknown" and table.loc["Adult", "count"] == 1
    assert table.loc["<null>", "kind"] == "null_value" and table.loc["<null>", "count"] == 3
    assert table["count"].sum() == 1806


def test_finished_run_is_write_once(finished, fake_shards, tmp_path) -> None:
    run, zip_path = finished
    watched = ["summary.json", "label_survey_report.md", "confusing_pairs.csv", "audit_pack.csv"]
    before = {n: hashlib.sha256((run.out_dir / n).read_bytes()).hexdigest() for n in watched}
    stamps = {n: (run.out_dir / n).stat().st_mtime_ns for n in watched}
    zip_path.unlink()
    again = make_run(fake_shards["directory"], tmp_path / "out")
    assert again.complete and again.run_id == run.run_id
    rebuilt = again.run_all()
    assert rebuilt.is_file()  # only the zip is rebuilt
    assert {n: (run.out_dir / n).stat().st_mtime_ns for n in watched} == stamps
    assert {n: hashlib.sha256((run.out_dir / n).read_bytes()).hexdigest() for n in watched} == before
    # steps still show their stored results instead of recomputing
    again.step_confusion()
    assert again.summary["model"]["n_docs"] == run.summary["model"]["n_docs"]


def test_interrupted_run_continues_with_the_missing_steps(finished, fake_shards, tmp_path) -> None:
    run, _ = finished
    keep = run.out_dir / "confusing_pairs.csv"
    stamp = keep.stat().st_mtime_ns
    for name in ("_COMPLETE.json", "summary.json", "label_survey_report.md",
                 "steps/scenarios.json", "steps/audit.json", "label_scenarios.csv", "audit_pack.csv"):
        (run.out_dir / name).unlink()
    resumed = make_run(fake_shards["directory"], tmp_path / "out")
    assert not resumed.complete
    resumed.run_all()
    assert (run.out_dir / "label_scenarios.csv").is_file() and (run.out_dir / "audit_pack.csv").is_file()
    assert keep.stat().st_mtime_ns == stamp  # finished steps were not redone
    assert report.is_complete(run.out_dir)


def test_bronze_ids_are_excluded_and_compared(fake_shards, tmp_path) -> None:
    bronze_ids = {f"doc-0-{g}-{r}" for g in range(4) for r in range(0, 150, 3)}
    counts = {"Sports": 30, "News": 20}
    bronze = lake_inputs.BronzeInfo("source_manifest_x.json", 50, "a" * 64, bronze_ids, counts)
    run = make_run(fake_shards["directory"], tmp_path / "out", bronze=bronze)
    run.run_all()
    sample = run.sample
    assert not bronze_ids & set(sample["id"])
    info = run.summary["survey_sample"]
    assert info["bronze_excluded"] is True and info["excluded_bronze_ids"] > 0
    assert run.summary["bronze_vs_full"]["status"] == "ok"
    assert (run.out_dir / "bronze_vs_full.csv").is_file()
    assert (run.out_dir / "fig_bronze_vs_full.png").is_file()
    # a different Bronze snapshot is a different run (different sample, never mixed up)
    other = make_run(fake_shards["directory"], tmp_path / "out")
    assert other.run_id != run.run_id and other.sample_cache != run.sample_cache


def test_existing_snapshots_are_never_touched(fake_shards, tmp_path) -> None:
    lake = tmp_path / "lake"
    for sub in ("01_bronze", "02_silver", "03_gold", "experiments"):
        (lake / sub).mkdir(parents=True)
        (lake / sub / "keep.txt").write_text("x", encoding="utf-8")
    run = make_run(fake_shards["directory"], lake / "04_reports" / "label_survey")
    run.run_all()
    for sub in ("01_bronze", "02_silver", "03_gold", "experiments"):
        assert [p.name for p in (lake / sub).iterdir()] == ["keep.txt"]


# ------------------------------------------------------------------ Data Lake inputs


def _write_bronze(lake: Path, name_suffix: str, rows: list[dict]) -> str:
    (lake / "01_bronze").mkdir(parents=True, exist_ok=True)
    (lake / "00_source_metadata").mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    file_name = f"vtsnlp_raw_sample_{name_suffix}.jsonl"
    (lake / "01_bronze" / file_name).write_bytes(data)
    manifest = {"file_name": file_name, "sha256": hashlib.sha256(data).hexdigest(),
                "sample_size": len(rows), "file_size_bytes": len(data)}
    manifest_name = f"source_manifest_{name_suffix}.json"
    (lake / "00_source_metadata" / manifest_name).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_name


def test_load_bronze_reads_ids_and_counts_with_the_existing_reader(tmp_path) -> None:
    rows = [{"id": "a", "text": "t", "domain": "News"}, {"id": "b", "text": "t", "domain": None}]
    name = _write_bronze(tmp_path, "b81fcce589_seed42_buf10000_n50000", rows)
    other = _write_bronze(tmp_path, "b81fcce589_seed1_buf10000_n20", rows)
    assert lake_inputs.find_bronze_manifest(tmp_path) == name  # prefers 50k / seed 42
    assert lake_inputs.find_bronze_manifest(tmp_path, other) == other
    info = lake_inputs.load_bronze(tmp_path, log=lambda *_: None)
    assert info.ids == {"a", "b"} and info.label_counts == {"News": 1, "<null>": 1}
    with pytest.raises(FileNotFoundError, match="USE_DRIVE"):
        lake_inputs.find_bronze_manifest(tmp_path / "empty-lake")
    with pytest.raises(FileNotFoundError, match="nope"):
        lake_inputs.find_bronze_manifest(tmp_path, "nope.json")


def test_snapshot_counts_report_unreadable_snapshots_instead_of_crashing(tmp_path) -> None:
    assert lake_inputs.snapshot_label_counts(tmp_path, log=lambda *_: None) == {}
    reports = tmp_path / "04_reports"
    reports.mkdir()
    (reports / "silver_quality_v1_broken.json").write_text("{}", encoding="utf-8")
    result = lake_inputs.snapshot_label_counts(tmp_path, log=lambda *_: None)
    (entry,) = result.values()
    assert entry["status"] == "unavailable" and "SilverReportError" in entry["reason"]
