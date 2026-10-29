"""Train one TF-IDF baseline from an existing immutable Gold snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vietnamese_topic_classification.config.settings import DataLakeConfig
from vietnamese_topic_classification.training import (
    ClassifierConfig,
    ExperimentConfig,
    FeatureConfig,
)
from vietnamese_topic_classification.training.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and validate TF-IDF multiclass baseline")
    parser.add_argument("--manifest-name", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--data-lake-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--classifier", choices=["logistic_regression", "linear_svc"], default="logistic_regression")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--max-features", type=int, default=100000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.min_df < 1 or args.max_features < 1:
        parser.error("--min-df and --max-features must be positive")
    config = ExperimentConfig(
        name=args.name,
        gold_manifest_name=args.manifest_name,
        seed=args.seed,
        features=FeatureConfig(params={"ngram_range": (1, 2), "min_df": args.min_df, "max_features": args.max_features}),
        classifier=ClassifierConfig(
            kind=args.classifier,
            params={"class_weight": None if args.class_weight == "none" else "balanced"},
        ),
    )
    result = run_experiment(
        config,
        data_lake_root=args.data_lake_root or DataLakeConfig.from_env().root,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
