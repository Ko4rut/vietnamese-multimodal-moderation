"""Build Gold V3 train/validation/test splits from a verified Silver V3 snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vietnamese_topic_classification.data.gold import GoldConfig, GoldReader, GoldV3Pipeline
from vietnamese_topic_classification.data.silver import SilverReader
from vietnamese_topic_classification.data.silver.v3 import SilverV3Pipeline


def build(args: argparse.Namespace) -> None:
    silver = SilverReader(args.data_lake_root, args.silver_v3_report)
    if silver.report.get("pipeline_version") != "v3":
        raise ValueError("Gold V3 requires a Silver V3 quality report")
    SilverV3Pipeline.verify_reader(silver)
    pipeline = GoldV3Pipeline(
        silver,
        config=GoldConfig(
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        ),
    )
    result = pipeline.run()
    gold = GoldReader(args.data_lake_root, Path(result["paths"]["manifest"]).name)
    print(json.dumps({
        "status": "published",
        "paths": result["paths"],
        "split_rows": result["report"]["split_rows"],
        "verified": gold.verify(),
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Gold V3 from Silver V3")
    parser.add_argument("--data-lake-root", type=Path, required=True)
    parser.add_argument("--silver-v3-report", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if Path(args.silver_v3_report).name != args.silver_v3_report:
        parser.error("--silver-v3-report must be a report filename, not a path")
    build(args)


if __name__ == "__main__":
    main()
