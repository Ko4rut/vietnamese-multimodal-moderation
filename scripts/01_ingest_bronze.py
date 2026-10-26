"""Run pinned VTSNLP Bronze ingestion after mounting the correct Google Drive."""

import argparse
import json
from pathlib import Path

from vietnamese_topic_classification.data.bronze import BronzeConfig, BronzeIngestor


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the pinned VTSNLP Bronze snapshot")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--local-temp-dir", type=Path, default=None)
    args = parser.parse_args()
    config = BronzeConfig(
        output_root=args.output_root,
        sample_size=args.sample_size,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
        local_temp_dir=args.local_temp_dir,
    )
    result = BronzeIngestor(config).run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
