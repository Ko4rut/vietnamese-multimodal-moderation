"""Offline contract checks for the Gold V2 Colab runner."""

from __future__ import annotations

import ast
import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/v2/colab_gold_v2_runner.ipynb"


def test_gold_v2_notebook_contract():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    ids = [cell["id"] for cell in nb["cells"]]
    assert ids == [
        "intro", "setup", "mount", "select-silver-v2",
        "preflight", "publish", "verify", "finish",
    ]
    sources = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse(cell["source"], filename=cell["id"])
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            sources.append(cell["source"])
    code = "\n".join(sources)
    for token in (
        '"VTC_GIT_REF", "main"', "drive.mount(MOUNT)",
        "resolve_mounted_lake(MOUNT)", "GoldV2Pipeline(",
        "pipeline.verify_silver_publication()", "GoldReader(",
        "gold_v2_reader.verify()", "PUBLISH_GOLD_V2 = False",
        "stale_uploads", "len(existing) != len(expected)",
    ):
        assert token in code
    assert "SilverV2Pipeline(" not in code
    assert "01_ingest_bronze.py" not in code
