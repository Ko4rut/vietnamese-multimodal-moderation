"""Explicit, write-once final evaluation of a selected trained model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from vietnamese_topic_classification.config import get_domain_labels
from vietnamese_topic_classification.config.settings import DataLakeConfig
from vietnamese_topic_classification.data.gold import GoldReader
from vietnamese_topic_classification.evaluation.metrics import calculate_metrics
from vietnamese_topic_classification.evaluation.report import write_evaluation_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a selected model on Gold test (one time)")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--data-lake-root", type=Path)
    args = parser.parse_args()
    artifact = args.artifact_dir
    metadata = json.loads((artifact / "metadata.json").read_text(encoding="utf-8"))
    config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    root = args.data_lake_root or DataLakeConfig.from_env().root
    reader = GoldReader(root, config["gold_manifest_name"])
    reader.verify()
    import hashlib

    manifest_hash = hashlib.sha256(reader.manifest_path.read_bytes()).hexdigest()
    if metadata["lineage"].get("gold_manifest_sha256") != manifest_hash:
        raise ValueError("Gold manifest does not match the training snapshot")
    report_path = artifact / "test_metrics.json"
    if report_path.exists():
        raise FileExistsError(f"Test evaluation already exists: {report_path}")
    rows = list(reader.iter_split("test"))
    if not rows:
        raise ValueError("Gold test split is empty")
    model = joblib.load(artifact / "model.joblib")
    predictions = model.predict([row["text"] for row in rows])
    metrics = calculate_metrics(
        [row["label_id"] for row in rows], predictions, label_names=get_domain_labels()
    )
    write_evaluation_report(metrics, output_path=report_path)
    print(json.dumps({"test_metrics_path": str(report_path), "macro_f1": metrics["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
