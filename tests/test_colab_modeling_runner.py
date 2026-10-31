"""Validate Colab modeling notebook without Colab, Drive or model dependencies."""

import ast
import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/v1/colab_modeling_runner.ipynb"


def test_colab_modeling_runner_contract():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    cells = notebook["cells"]
    assert len(cells) >= 8
    assert len({cell["id"] for cell in cells}) == len(cells)
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    for cell in code_cells:
        ast.parse(cell["source"])
        assert cell["outputs"] == []
        assert cell["execution_count"] is None

    source = "\n".join(cell["source"] for cell in code_cells)
    for required in (
        'BRANCH = __import__("os").environ.get("VTC_GIT_REF", "main")',
        'DATA_LAKE_ROOT',
        '"03_gold"',
        'GoldReader(',
        'reader.verify()',
        '"--data-lake-root"',
        '"--manifest-name"',
        '"--classifier"',
        '"--class-weight"',
        '"--max-features"',
        'scripts/06_train_baseline.py',
        '"logistic_regression"',
        '"linear_svc"',
        'macro_f1',
        'predict_proba',
        'joblib.load',
        'verify_existing_experiment',
        'manifest_sha256',
        'RUN_BALANCED = False',
        'scripts/08_plot_diagnostics.py',
        'diagnostics_v1',
        'train_vs_validation.png',
        'correct_counts_by_class.png',
    ):
        assert required in source
    assert "scripts/07_evaluate_model.py" not in source
    assert 'iter_split("test")' not in source
    assert '"04_gold"' not in source
    assert '01_ingest_bronze.py' not in source
