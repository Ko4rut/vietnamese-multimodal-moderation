"""Validate the Bronze News EDA notebook and run its non-Colab cells on a fake Data Lake."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")
pytest.importorskip("yaml")
nbformat = pytest.importorskip("nbformat")
pytest.importorskip("IPython")

import bronze_news_fakes as fakes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/16_generate_bronze_news_eda_notebook.py"
NOTEBOOK = ROOT / "notebooks/v3/eda/01_bronze_news_eda_colab.ipynb"


def load_cells() -> list[dict]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def tree_hashes(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_generated_notebook_is_current(tmp_path) -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--check"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    outdated = tmp_path / "old.ipynb"
    outdated.write_text("outdated", encoding="utf-8")
    result = subprocess.run([sys.executable, str(SCRIPT), "--check", "--output", str(outdated)],
                            capture_output=True)
    assert result.returncode != 0
    fresh = tmp_path / "fresh.ipynb"
    subprocess.run([sys.executable, str(SCRIPT), "--output", str(fresh)], check=True)
    assert fresh.read_text(encoding="utf-8") == NOTEBOOK.read_text(encoding="utf-8")


def test_notebook_is_valid_and_clean() -> None:
    nbformat.validate(nbformat.read(NOTEBOOK, as_version=4))
    cells = load_cells()
    ids = [c["id"] for c in cells]
    assert len(ids) == len(set(ids))
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None and cell["outputs"] == []
        ast.parse(cell["source"])
        assert "input(" not in cell["source"] and "getpass" not in cell["source"]
        assert not re.search(r"^\s*[!%]", cell["source"], re.M), "no shell/magic lines"


def test_cell_order_tags_and_no_lake_writes() -> None:
    cells = load_cells()
    order = [c["id"] for c in cells]
    assert order.index("config") < order.index("clone") < order.index("lake") < order.index("section-1")
    tagged = [c["id"] for c in cells if "colab-only" in c["metadata"].get("tags", [])]
    assert tagged == ["clone", "install", "drive"]
    sections = [c["id"] for c in cells if c["id"].startswith("section-") and c["cell_type"] == "code"]
    assert len(sections) >= 7
    source = "\n".join(c["source"] for c in cells if c["cell_type"] == "code")
    for token in ("BronzeNewsConfig", "load_bronze_news", "fit_feature_space", "diagnose",
                  "run_comparison", "neighbor_overlap", "export_results", "choose_output_root"):
        assert token in source
    assert "04_reports" not in source and "DATA_LAKE_ROOT /" not in source
    assert "3006" not in source  # the News count is measured, never hard-coded


def test_config_cell_defaults() -> None:
    cells = {c["id"]: c for c in load_cells()}
    ns: dict = {}
    exec(cells["config"]["source"], ns)
    assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", ns["GIT_REF"])
    assert ns["GIT_REF"] == "feat/bronze-news-eda-v3"
    expected = {"SEED": 42, "USE_DRIVE": True, "BRONZE_MANIFEST_NAME": None,
                "OUTPUT_ROOT_OVERRIDE": None, "DATA_LAKE_ROOT_OVERRIDE": None,
                "EXPORT_RESULTS": True, "AUTO_DOWNLOAD_ZIP": True,
                "INCLUDE_NEIGHBOR_LABELS": True, "DROP_EXACT_DUPLICATES": True}
    for key, value in expected.items():
        assert ns[key] == value, key
    from vietnamese_topic_classification.analysis.bronze_news.config import BronzeNewsConfig

    cfg = BronzeNewsConfig.from_mapping(ns)
    assert cfg.seed == 42 and len(cfg.algorithms) == 13


def run_cells(lake: Path, out: Path) -> dict:
    namespace: dict = {"PROJECT_DIR": None}
    for cell in load_cells():
        if cell["cell_type"] != "code" or "colab-only" in cell["metadata"].get("tags", []):
            continue
        exec(compile(cell["source"], cell["id"], "exec"), namespace)
        if cell["id"] == "config":
            namespace.update(
                DATA_LAKE_ROOT_OVERRIDE=str(lake), OUTPUT_ROOT_OVERRIDE=str(out), USE_DRIVE=False,
                MIN_DF=2, SVD_COMPONENTS=30, K_RANGE=(3, 8), NEIGHBOR_PER_LABEL=40,
                TSNE_MAX_POINTS=500, NEAR_DUP_SAMPLE=500, SLOW_MAX_POINTS=200,
                KEYWORDS_PER_CLUSTER=15,
            )
            namespace["MOUNTED"] = False   # the Colab-only drive cell is skipped here
    return namespace


def test_notebook_cells_run_end_to_end_on_fake_lake(tmp_path, capsys) -> None:
    lake, out = tmp_path / "lake", tmp_path / "out"
    info = fakes.write_lake(lake)
    lake_before = tree_hashes(lake)
    ns = run_cells(lake, out)
    printed = capsys.readouterr().out
    assert tree_hashes(lake) == lake_before  # nothing written to the Data Lake

    run_dir = out / "bronze_news_eda" / ns["run_id"]
    assert (run_dir / "_COMPLETE.json").is_file()
    for name in ("cluster_assignments.csv", "cluster_summary.csv", "cluster_summary.json",
                 "algorithm_comparison.csv", "eda_summary.json", "report.md", "run_config.json",
                 "cluster_excerpts_INTERNAL.csv"):
        assert (run_dir / name).is_file(), name
    figures = {p.name for p in (run_dir / "figures").glob("*.png")}
    assert {"eda_length_hist.png", "tfidf_svd_variance.png", "shape_projection.png",
            "cluster_algorithm_grid.png", "cluster_neighbor_heatmap.png",
            "cluster_neighbor_scatter.png", "cluster_sizes.png"} <= figures
    assert (out / "bronze_news_eda" / f"{ns['run_id']}.zip").is_file()

    # measured, printed counts (not hard-coded)
    assert f"Số bài nhãn 'News': {4 * 90 + 7:,}" in printed
    assert info["sha256"] in printed and "Cấu hình phân tích sâu" in printed
    assert "BỎ QUA" in printed or "bỏ qua" in printed.lower()
    assert "Không tự tải được file zip" in printed  # google.colab absent: guidance, no crash

    import pandas as pd

    assign = pd.read_csv(run_dir / "cluster_assignments.csv")
    assert len(assign) == 362 and {"id", "tsne_x", "distance_to_centroid"} <= set(assign.columns)
    summary = json.loads((run_dir / "cluster_summary.json").read_text(encoding="utf-8"))
    assert len(summary) >= 3 and all(r["keywords"] and "nearest_neighbor_label" in r for r in summary)
    assert all(r["hint_note"].startswith("gợi ý") for r in summary)
    texts = [r["text"] for r in info["rows"] if r["text"] and r["text"].strip()]
    probes = [t.strip()[:60] for t in texts][:150]
    for path in run_dir.rglob("*"):
        if path.is_file() and path.suffix != ".png" and path.name != "cluster_excerpts_INTERNAL.csv":
            content = path.read_text(encoding="utf-8")
            assert not any(p in content for p in probes), path.name

    # Run all again: the complete directory is never overwritten.
    before = tree_hashes(run_dir)
    ns2 = run_cells(lake, out)
    printed2 = capsys.readouterr().out
    assert ns2["run_id"] == ns["run_id"] and "KHÔNG ghi đè" in printed2
    assert tree_hashes(run_dir) == before
