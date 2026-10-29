"""Immutable, per-experiment sklearn artifact publication."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn

from vietnamese_topic_classification.data.gold import GoldReader
from vietnamese_topic_classification.evaluation.report import write_evaluation_report

from .config import ExperimentConfig
from .model import ExperimentResult


def save_artifacts(
    model: object,
    result: ExperimentResult,
    config: ExperimentConfig,
    *,
    output_root: Path,
    data_lake_root: Path | None = None,
) -> Path:
    """Publish fitted pipeline and metadata once; never replace a prior experiment."""
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / config.name
    if destination.exists():
        raise FileExistsError(f"Experiment artifacts already exist: {destination}")
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=output_root))
    try:
        joblib.dump(model, staging / "model.joblib")
        effective_config = {
            **asdict(config),
            "features": {"kind": config.features.kind, "params": dict(config.features.params)},
            "classifier": {
                "kind": config.classifier.kind,
                "params": dict(config.classifier.params),
            },
        }
        (staging / "config.json").write_text(
            json.dumps(effective_config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        lineage: dict[str, object] = {"gold_manifest_name": config.gold_manifest_name}
        if data_lake_root is not None:
            reader = GoldReader(data_lake_root, config.gold_manifest_name)
            reader.verify()
            lineage["gold_manifest_sha256"] = hashlib.sha256(
                reader.manifest_path.read_bytes()
            ).hexdigest()
            lineage["gold_source"] = reader.manifest["source"]
        metadata = {
            "experiment_name": config.name,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "sklearn_version": sklearn.__version__,
            "seed": config.seed,
            "selection_metric": config.selection_metric,
            "lineage": lineage,
            "validation_metrics": result["validation_metrics"],
            "test_evaluated": False,
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if "validation_report" in result:
            write_evaluation_report(
                result["validation_report"], output_path=staging / "validation_metrics.json"
            )
        else:
            (staging / "validation_metrics.json").write_text(
                json.dumps(result["validation_metrics"], indent=2) + "\n",
                encoding="utf-8",
            )
        if destination.exists():
            raise FileExistsError(f"Experiment artifacts already exist: {destination}")
        staging.rename(destination)
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging)
