"""Check the Colab Gold runner without requiring Google Drive or notebook dependencies."""

import ast
import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/colab_gold_runner.ipynb"


def test_gold_colab_runner_contract():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    cells = notebook["cells"]
    assert len(cells) >= 7
    assert len({cell["id"] for cell in cells}) == len(cells)
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    for cell in code_cells:
        ast.parse(cell["source"])
        assert cell["outputs"] == []
        assert cell["execution_count"] is None

    source = "\n".join(cell["source"] for cell in code_cells)
    assert 'BRANCH = __import__("os").environ.get("VTC_GIT_REF", "main")' in source
    assert 'SILVER_REPORT_NAME = "silver_quality_v1_b81fcce589_seed42_buf10000_n50000.json"' in source
    for expected in (
        "drive.mount(MOUNT)",
        "SilverReader(",
        "silver.verify()",
        "gold.planned_paths(",
        ".uploading",
        "GoldReader(",
        "existing_reader.verify()",
        "scripts/05_build_gold.py",
        '"--data-lake-root"',
        '"--report-name"',
        "reader.iter_split(split)",
        "reader.verify()",
        "manifest",
        "silver_verified",
    ):
        assert expected in source
    assert "01_ingest_bronze.py" not in source
    assert '"04_gold"' in source
    assert 'GOLD_ALREADY_EXISTS' in source
    assert '"03_gold"' not in source
    assert "02_build_silver.py" not in source
    assert 'pip", "install", "-q", "-e", "."' in source
