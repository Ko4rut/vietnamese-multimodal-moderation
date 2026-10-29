"""Publish a Gold V2 snapshot from an existing, audited Silver V2 quality report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vietnamese_topic_classification.config.settings import DataLakeConfig
from vietnamese_topic_classification.data.gold.config import GoldConfig
from vietnamese_topic_classification.data.gold.pipeline_v2 import GoldV2Pipeline
from vietnamese_topic_classification.data.silver import SilverReader


def main() -> None:
    parser = argparse.ArgumentParser(description="Build immutable Gold V2 from Silver V2")
    parser.add_argument("--report-name", required=True)
    parser.add_argument("--data-lake-root", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--similarity-threshold", type=float, default=0.9)
    parser.add_argument("--max-candidates", type=int, default=2000)
    parser.add_argument(
        "--allow-unreviewed", action="store_true",
        help="Explicitly accept Silver V2 review/guardrail records without manual approval",
    )
    args = parser.parse_args()
    root = args.data_lake_root or DataLakeConfig.from_env().root
    reader = SilverReader(root, args.report_name)
    config = GoldConfig(
        seed=args.seed, similarity_threshold=args.similarity_threshold,
        max_candidates_per_record=args.max_candidates,
    )
    result = GoldV2Pipeline(
        reader, config=config, allow_unreviewed=args.allow_unreviewed
    ).run()
    print(json.dumps({
        "paths": result["paths"], "split_rows": result["report"]["split_rows"],
        "near_duplicate_audit": result["report"]["near_duplicate_audit"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
