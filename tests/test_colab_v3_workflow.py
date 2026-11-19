"""Offline contract checks for the stage-specific V3 Colab notebooks."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "notebooks" / "v3"
NOTEBOOKS = {
    name: ROOT / f"colab_{name}_v3_runner.ipynb"
    for name in ("silver", "gold", "modeling")
}
NOTEBOOKS["end_to_end"] = ROOT / "colab_end_to_end_v3.ipynb"


def _read_notebook(path: Path) -> tuple[dict[str, object], list[str], str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    ids = [cell["id"] for cell in notebook["cells"]]
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    sources = []
    for cell in code:
        source = cell["source"]
        source = source if isinstance(source, str) else "".join(source)
        ast.parse(source, filename=f"{path.name}:{cell['id']}")
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        sources.append(source)
    return notebook, ids, "\n".join(sources)


def test_v3_colab_notebooks_follow_stage_contracts() -> None:
    notebooks = {
        key: _read_notebook(path)
        for key, path in NOTEBOOKS.items()
    }
    assert not (ROOT / "colab_silver_to_gold_v3.ipynb").exists()

    silver_ids, silver_code = notebooks["silver"][1:]
    assert silver_ids == [
        "intro", "setup", "mount", "select-source", "keyword-analysis",
        "pilot-sample", "local-model-guide", "local-model-run", "review-template",
        "publish-silver", "layout",
    ]
    for token in (
        "resolve_mounted_lake(MOUNT)",
        "build_keyword_report(silver_v2.iter_records())",
        "PILOT_SAMPLE_SIZE = 300",
        "RUN_LOCAL_MODEL_LABELING = False",
        "scripts/15_label_news_v3_local.py",
        "CREATE_FULL_REVIEW_TEMPLATE = False",
        "SilverV3Pipeline",
        "PUBLISH_SILVER_V3 = False",
    ):
        assert token in silver_code
    assert "PUBLISH_GOLD_V3" not in silver_code
    assert "RUN_MODELING_V3" not in silver_code

    gold_ids, gold_code = notebooks["gold"][1:]
    assert gold_ids == ["intro", "setup", "mount", "build-gold-v3", "outputs"]
    for token in (
        "silver_quality_v3_",
        "SilverV3Pipeline.verify_reader",
        "GoldV3Pipeline",
        "PUBLISH_GOLD_V3 = False",
        "stale_gold",
        "GoldReader",
    ):
        assert token in gold_code
    assert "GoldV2Pipeline(" not in gold_code

    model_ids, model_code = notebooks["modeling"][1:]
    assert model_ids == ["intro", "setup", "mount", "select-gold-v3", "train-compare-v3", "outputs"]
    for token in (
        "gold_manifest_v3_",
        "gold_v3.verify()",
        "len(gold_v3.manifest[\"label_to_id\"]) != 27",
        "tfidf_baselines_v3",
        "run_experiment(",
        "test_evaluated",
        "Gold test was verified from checksums, not loaded or evaluated.",
    ):
        assert token in model_code
    assert "get_domain_labels()" not in model_code

    e2e_ids, e2e_code = notebooks["end_to_end"][1:]
    assert e2e_ids == [
        "intro", "setup", "mount", "select-source", "keyword-analysis",
        "pilot-sample", "local-model-guide", "local-model-run", "review-template",
        "publish-silver", "publish-gold", "modeling-v3", "layout",
    ]
    assert e2e_ids.index("publish-silver") < e2e_ids.index("publish-gold") < e2e_ids.index("modeling-v3")
    for token in ("PUBLISH_SILVER_V3 = False", "PUBLISH_GOLD_V3 = False", "RUN_MODELING_V3 = False"):
        assert token in e2e_code
    assert "GoldV2Pipeline(" not in e2e_code
    assert "gold_manifest_v2" not in e2e_code


def test_v3_notebook_dependencies_match_the_stage() -> None:
    for stage in ("silver", "gold"):
        _, _, source = _read_notebook(NOTEBOOKS[stage])
        assert '\"-e\", \".\"' in source
        assert '\"-e\", \".[modeling]\"' not in source
    for stage in ("modeling", "end_to_end"):
        _, _, source = _read_notebook(NOTEBOOKS[stage])
        assert '\"-e\", \".[modeling]\"' in source
