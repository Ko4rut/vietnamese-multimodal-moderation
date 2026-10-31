"""Offline structure and safety contract for the Gold V2 modeling Colab runner."""

from __future__ import annotations

import ast
import json
from pathlib import Path

NOTEBOOK = (
    Path(__file__).resolve().parents[1] / "notebooks/v2/colab_modeling_runner.ipynb"
)


def test_colab_modeling_v2_notebook_contract():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    ids = [cell["id"] for cell in nb["cells"]]
    assert ids == [
        "intro", "checkout", "install", "drive", "configuration",
        "train", "compare", "compare-v1", "diagnostics", "probabilities", "next",
    ]
    assert len(ids) == len(set(ids))

    code = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell["outputs"] == []
        assert cell["execution_count"] is None
        ast.parse(cell["source"], filename=cell["id"])
        code.append(cell["source"])
    source = "\n".join(code)
    for required in (
        '"VTC_GIT_REF", "main"',
        '"pip", "install", "-q", "-e", ".[modeling]"',
        "importlib.invalidate_caches()",
        "drive.mount(MOUNT)",
        "resolve_mounted_lake(MOUNT)",
        '"03_gold" / "v2" / "manifests"',
        '"tfidf_baselines_v2"',
        '"gold_manifest_v2_"',
        "GoldReader(",
        "reader.verify()",
        'reader.version != "v2"',
        'source.get("silver_version") != "v2"',
        'RUN_BALANCED = False',
        'RUN_DIAGNOSTICS = False',
        'manifest_sha256',
        'verify_existing_experiment',
        'metadata.get("lineage"',
        'scripts/06_train_baseline.py',
        '"logistic_regression"',
        '"linear_svc"',
        '"macro_f1"',
        '"validation_metrics.json"',
        "V1_ROOT",
        "compare-v1" if False else "V1 baseline",
        "if RUN_DIAGNOSTICS:",
    ):
        assert required in source
    assert '"03_gold" / "manifests"' not in source
    assert '"tfidf_baselines_v1"' in source  # Only read-only V1 report comparison.
    assert 'scripts/07_evaluate_model.py' not in source
    assert 'iter_split("test")' not in source
    assert '01_ingest_bronze.py' not in source
    assert '09_build_silver_v2.py' not in source
    assert "SilverV2Pipeline(" not in source
