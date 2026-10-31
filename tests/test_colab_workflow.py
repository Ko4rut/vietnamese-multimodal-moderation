"""Validate all team-facing Colab notebooks offline.

Colab OAuth and the real 50K download are deliberately not part of CI.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

NOTEBOOKS = [
    "colab_runner.ipynb",
    "colab_silver_runner.ipynb",
    "colab_silver_v2_runner.ipynb",
    "colab_gold_runner.ipynb",
    "colab_modeling_runner.ipynb",
    "colab_data_lake_end_to_end.ipynb",
]
ROOT = Path(__file__).resolve().parents[1] / "notebooks"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_colab_code_parses_without_stale_feature_branches(name):
    nb = json.loads((ROOT / name).read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert len({cell["id"] for cell in nb["cells"]}) == len(nb["cells"])
    code = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
    assert code
    for cell in code:
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        ast.parse(cell["source"], filename=f"{name}:{cell['id']}")
    source = "\n".join(cell["source"] for cell in code)
    assert '"VTC_GIT_REF", "main"' in source
    assert "drive.mount(MOUNT)" in source
    assert "resolve_mounted_lake(MOUNT)" in source
    assert "feat/gold-pipeline-v1" not in source
    assert "feat/silver-pipeline-v1" not in source
    assert "feat/silver-v2-boilerplate-cleaning" not in source
    assert "feat/modeling-diagnostics-v1" not in source


def test_end_to_end_orders_stages_and_verifies_existing():
    nb = json.loads((ROOT / "colab_data_lake_end_to_end.ipynb").read_text())
    ids = [cell["id"] for cell in nb["cells"]]
    assert ids == [
        "intro", "setup", "mount", "parameters", "bronze",
        "silver-v1", "silver-v2", "gold", "finish",
    ]
    cells = {cell["id"]: cell["source"] for cell in nb["cells"]}
    assert "BronzeReader(" in cells["bronze"]
    assert "bronze_reader.verify()" in cells["bronze"]
    assert "if bronze_existing:" in cells["bronze"]
    assert "silver_v1_pipeline.planned_paths()" in cells["silver-v1"]
    assert "v1_reader.verify()" in cells["silver-v1"]
    assert "silver_v2_pipeline.run(dry_run=True)" in cells["silver-v2"]
    assert "PUBLISH_SILVER_V2 = False" in cells["parameters"]
    assert "GOLD_SOURCE_VERSION = \"v1\"" in cells["parameters"]
    assert "gold_reader.verify()" in cells["gold"]
    assert "if existing_gold:" in cells["gold"]
    assert "GoldPipeline(" in cells["gold"]
    assert "GoldReader(" in cells["gold"]


def test_silver_v1_colab_does_not_use_stale_constructor_or_paths():
    nb = json.loads((ROOT / "colab_silver_runner.ipynb").read_text())
    source = "\n".join(cell["source"] for cell in nb["cells"] if cell["cell_type"] == "code")
    assert "BronzeReader(data_lake_root=DATA_LAKE_ROOT, metadata_name=MANIFEST_NAME)" in source
    assert "SilverPipeline(reader)" in source
    assert "pipeline.planned_paths()" in source
    assert "SILVER_ALREADY_EXISTS" in source
    assert "BronzeReader(drive_root=" not in source


def test_gold_and_v2_reference_new_silver_report_layout():
    for name in ("colab_gold_runner.ipynb", "colab_silver_v2_runner.ipynb"):
        nb = json.loads((ROOT / name).read_text())
        source = "\n".join(
            cell["source"] for cell in nb["cells"] if cell["cell_type"] == "code"
        )
        assert '"05_reports" / "silver" / "v1"' in source
    nb = json.loads((ROOT / "colab_silver_v2_runner.ipynb").read_text())
    source = "\n".join(cell["source"] for cell in nb["cells"] if cell["cell_type"] == "code")
    assert "from vietnamese_topic_classification.data.silver.v2.cleaner import" in source
    assert "from vietnamese_topic_classification.data.silver.v2.corpus import" in source
    assert "PUBLISH = False" in source
