"""Validate the Colab survey notebook and run its non-Colab cells on fake data."""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")
nbformat = pytest.importorskip("nbformat")
pytest.importorskip("IPython")

from vietnamese_topic_classification.analysis.label_survey import runner as runner_module  # noqa: E402
from vietnamese_topic_classification.analysis.label_survey.config import SurveyConfig  # noqa: E402
from vietnamese_topic_classification.analysis.label_survey.hf_parquet import (  # noqa: E402
    LocalParquetSource,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/11_generate_label_survey_notebook.py"
NOTEBOOK = ROOT / "notebooks/v2/eda/02_label_survey_colab.ipynb"


def load_cells() -> list[dict]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def test_generated_notebook_is_current(tmp_path) -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--check"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    outdated = tmp_path / "old.ipynb"
    outdated.write_text("outdated", encoding="utf-8")
    result = subprocess.run([sys.executable, str(SCRIPT), "--check", "--output", str(outdated)],
                            capture_output=True)
    assert result.returncode != 0


def test_notebook_is_valid_and_clean() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    cells = load_cells()
    ids = [c["id"] for c in cells]
    assert len(ids) == len(set(ids))
    code = [c for c in cells if c["cell_type"] == "code"]
    for cell in code:
        assert cell["execution_count"] is None and cell["outputs"] == []
        ast.parse(cell["source"])
        assert "input(" not in cell["source"]
        assert not re.search(r"^\s*[!%]", cell["source"], re.M), "no shell/magic lines"
    for cell in cells:
        assert cell["cell_type"] != "code" or "getpass" not in cell["source"]


def test_config_cell_matches_the_plan_and_the_dataclass() -> None:
    cells = {c["id"]: c for c in load_cells()}
    assert cells["config"]["id"] == "config"
    order = [c["id"] for c in load_cells()]
    assert order.index("config") < order.index("clone") < order.index("section-4")
    namespace: dict = {}
    exec(cells["config"]["source"], namespace)
    assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", namespace["GIT_REF"])
    assert namespace["GIT_REF"] == "feat/label-survey"
    expected = {"SURVEY_TARGET_ROWS": 300_000, "MAX_DOWNLOAD_GB": 4.0, "PER_CLASS_CAP": 4_000,
                "TEXT_CHAR_LIMIT": 5_000, "HEAD_CHARS": 1_000, "N_FOLDS": 5, "TOP_PAIRS": 15,
                "MISMATCH_PROB": 0.70, "AUDIT_PER_CLASS": 12, "AUDIT_PER_PAIR": 10,
                "RUN_HEAD_VS_FULL": True, "AUTO_DOWNLOAD_ZIP": True, "USE_DRIVE": True,
                "BRONZE_MANIFEST_NAME": None, "SEED": 42}
    for key, value in expected.items():
        assert namespace[key] == value, key
    config = SurveyConfig.from_mapping(namespace)
    assert config.hf_revision == namespace["HF_REVISION"] and config.n_folds == 5
    assert config.audit_part_c == 30


def test_only_the_expected_cells_need_colab() -> None:
    tagged = [c["id"] for c in load_cells() if "colab-only" in c["metadata"].get("tags", [])]
    assert tagged == ["clone", "install", "drive", "bronze"]


def test_notebook_cells_run_end_to_end_on_fake_data(fake_shards, tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(runner_module, "HfParquetSource",
                        lambda dataset, revision: LocalParquetSource(fake_shards["directory"]))
    overrides = dict(
        SURVEY_TARGET_ROWS=1200, PER_CLASS_CAP=150, N_FOLDS=3, TOP_PAIRS=3, LR_MAX_ITER=100,
        HTTP_WORKERS=4, N_JOBS=1, MAX_FEATURES=2000, AUDIT_PER_CLASS=3, AUDIT_PER_PAIR=4,
        AUDIT_PART_C=5, HEAD_CHARS=200, TEXT_CHAR_LIMIT=500,
    )
    namespace: dict = {
        "PROJECT_DIR": None, "OUTPUT_ROOT": tmp_path / "out", "bronze": None, "snapshots": {},
        "BRONZE_SKIP_REASON": "kiểm thử không có Data Lake",
    }
    for cell in load_cells():
        if cell["cell_type"] != "code" or "colab-only" in cell["metadata"].get("tags", []):
            continue
        exec(compile(cell["source"], cell["id"], "exec"), namespace)
        if cell["id"] == "config":
            namespace.update(overrides)   # shrink the run; nothing else is changed
    run = namespace["run"]
    assert run.complete and namespace["zip_path"].is_file()
    assert (run.out_dir / "label_survey_report.md").is_file()
    output = capsys.readouterr().out
    assert "HOÀN TẤT" in output and "label_survey_report.md" in output
    assert "Không tự tải được file zip" in output  # google.colab is absent: guidance, no crash
    assert run.summary["bronze_vs_full"]["reason"] == "kiểm thử không có Data Lake"
