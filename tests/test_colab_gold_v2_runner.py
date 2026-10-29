"""Validate Gold V2 Colab notebook without Drive access."""

import ast
import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/colab_gold_v2_runner.ipynb"


def test_gold_v2_colab_contract():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    code = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
    assert len({cell["id"] for cell in nb["cells"]}) == len(nb["cells"])
    for cell in code:
        ast.parse(cell["source"])
        assert cell["outputs"] == [] and cell["execution_count"] is None
    content = "\n".join(cell["source"] for cell in code)
    for term in (
        'BRANCH = "feat/gold-v2-cleaned-dataset"',
        'sys.path.insert(0, str(SRC_DIR))',
        'silver_quality_v2_*.json',
        'GoldV2Pipeline(',
        'scripts/10_build_gold_v2.py',
        'GoldReader(',
        'reader.verify()',
        'ALLOW_UNREVIEWED = False',
        '"--allow-unreviewed"',
    ):
        assert term in content
    assert "01_ingest_bronze.py" not in content
