"""Create post-fit charts beside an existing Gold TF-IDF experiment (no test set)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib

from vietnamese_topic_classification.config import get_domain_labels
from vietnamese_topic_classification.config.settings import DataLakeConfig
from vietnamese_topic_classification.evaluation.diagnostics import save_diagnostics
from vietnamese_topic_classification.training.dataset import load_gold_training_splits
from vietnamese_topic_classification.data.gold import GoldReader


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot train/validation gaps, training-size learning curves and class performance"
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--data-lake-root", type=Path)
    args = parser.parse_args()

    artifact_dir = args.artifact_dir.resolve()
    root = args.data_lake_root or DataLakeConfig.from_env().root
    output_dir = artifact_dir / "diagnostics_v1"
    if output_dir.exists():
        print(f"[diagnostics] Existing output: {output_dir} (skip; write-once)", flush=True)
        return

    config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    if config.get("name") != metadata.get("experiment_name"):
        raise ValueError("Experiment name mismatch in saved artifacts")
    reader = GoldReader(root, config["gold_manifest_name"])
    reader.verify()
    actual_sha = hashlib.sha256(reader.manifest_path.read_bytes()).hexdigest()
    if metadata.get("lineage", {}).get("gold_manifest_sha256") != actual_sha:
        raise ValueError("Gold manifest SHA256 does not match the trained model")
    train, validation = load_gold_training_splits(
        data_lake_root=root, manifest_name=config["gold_manifest_name"]
    )
    # Only load models created by your own trusted training pipeline: joblib executes pickle.
    pipeline = joblib.load(artifact_dir / "model.joblib")
    report = save_diagnostics(
        pipeline, train, validation, label_names=get_domain_labels(),
        output_dir=output_dir, seed=config["seed"],
    )
    print(
        json.dumps(
            {"charts": str(output_dir), "train_vs_validation_gap": report["train_minus_validation_gap"]},
            ensure_ascii=False, indent=2,
        )
    )


if __name__ == "__main__":
    main()
