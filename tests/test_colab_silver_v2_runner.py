"""Static offline checks for the Silver V2 Colab runner.

Avoid recurring editable-install import regressions in Colab notebooks.
"""

import ast
import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/colab_silver_v2_runner.ipynb"


def test_colab_silver_v2_runner_import_and_execution_contract():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    cells = nb["cells"]
    assert len({cell["id"] for cell in cells}) == len(cells)
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    for cell in code_cells:
        ast.parse(cell["source"])
        assert cell["execution_count"] is None
        assert cell["outputs"] == []

    setup = next(cell["source"] for cell in code_cells if cell["id"] == "setup")
    preflight = next(cell["source"] for cell in code_cells if cell["id"] == "preflight")
    assert 'BRANCH = "feat/silver-v2-boilerplate-cleaning"' in setup
    assert '"pip", "install", "-q", "-e", "."' in setup
    for source in (setup, preflight):
        assert '(PROJECT_DIR / "src").resolve()' in source
        assert 'sys.path.insert(0, str(SRC_DIR))' in source
        assert "from vietnamese_topic_classification.data.silver.pipeline_v2" in source
    assert "importlib.invalidate_caches()" in setup
    source = "\n".join(cell["source"] for cell in code_cells)
    assert "SilverReader(" in source
    assert "reader.verify()" in source
    assert "dry_run=True" in source
    assert "PUBLISH = False" in source
    assert "expected[\"report\"]" in source
    assert "01_ingest_bronze.py" not in source
