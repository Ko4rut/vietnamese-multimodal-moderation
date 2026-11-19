"""Analyze Silver V2 News and publish Silver V3 from reviewed decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from vietnamese_topic_classification.analysis.news_v3 import (
    build_keyword_report,
    build_prompt,
    sample_news_records,
)
from vietnamese_topic_classification.config import get_v3_domain_labels
from vietnamese_topic_classification.data.gold.v2 import GoldV2Pipeline
from vietnamese_topic_classification.data.silver import SilverReader
from vietnamese_topic_classification.data.silver.v3 import SilverV3Pipeline


def _source(root: Path, report_name: str) -> SilverReader:
    if Path(report_name).name != report_name:
        raise ValueError("Silver report must be a filename, not a path")
    reader = SilverReader(root, report_name)
    if reader.report.get("pipeline_version") != "v2":
        raise ValueError("Silver V3 requires a Silver V2 quality report")
    GoldV2Pipeline(reader).verify_silver_publication()
    return reader


def analyze(args: argparse.Namespace) -> None:
    source = _source(args.data_lake_root, args.silver_v2_report)
    report = build_keyword_report(
        source.iter_records(),
        top_n=args.top_n,
        examples_per_domain=args.examples_per_domain,
    )
    report["source"] = {
        "silver_v2_report": source.report_path.name,
        "silver_v2_sha256": source.report["silver_sha256"],
    }
    output = args.output or (
        source.root / "04_reports" / "silver" / "v3"
        / (
            "news_keyword_analysis_v3_"
            f"{source.report['silver_sha256'][:12]}_"
            f"{_sha256(source.report_path)[:8]}.json"
        )
    )
    _write_new_json(output, report)
    print(json.dumps({
        "status": "analysis_saved",
        "path": str(output),
        "news_documents": report["classes"].get("News", {}).get("documents", 0),
        "classes": len(report["classes"]),
    }, ensure_ascii=False, indent=2))


def prepare(args: argparse.Namespace) -> None:
    source = _source(args.data_lake_root, args.silver_v2_report)
    count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        for row in source.iter_records():
            if row["domain"] != "News":
                continue
            handle.write(json.dumps({
                "id": row["id"],
                "source_domain": "News",
                "text": row["text"][:5000],
                "prompt": build_prompt(row, get_v3_domain_labels()),
                "label": "",
                "needs_review": True,
                "approved": False,
                "reviewer": "",
                "reason": "",
            }, ensure_ascii=False) + "\n")
            count += 1
    print(json.dumps({
        "status": "review_file_created",
        "news_documents": count,
        "output": str(args.output),
        "labels": get_v3_domain_labels(),
        "next": "Review each row; set label, approved=true, needs_review=false, reviewer and reason.",
    }, ensure_ascii=False, indent=2))


def pilot(args: argparse.Namespace) -> None:
    source = _source(args.data_lake_root, args.silver_v2_report)
    sample = sample_news_records(
        source.iter_records(),
        sample_size=args.sample_size,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        for row in sample:
            handle.write(json.dumps({
                "id": row["id"],
                "source_domain": "News",
                "text": str(row["text"])[:5000],
                "prompt": build_prompt(row, get_v3_domain_labels()),
                "pilot_only": True,
                "label": "",
                "needs_review": True,
                "approved": False,
                "reviewer": "",
                "reason": "",
            }, ensure_ascii=False) + "\n")
    print(json.dumps({
        "status": "pilot_file_created",
        "news_documents": len(sample),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "output": str(args.output),
        "note": "Pilot decisions do not cover all News rows and cannot publish Silver V3.",
    }, ensure_ascii=False, indent=2))


def build(args: argparse.Namespace) -> None:
    source = _source(args.data_lake_root, args.silver_v2_report)
    result = SilverV3Pipeline(source, args.decisions).run()
    published = SilverReader(args.data_lake_root, Path(result["paths"]["report"]).name)
    verified = SilverV3Pipeline.verify_reader(published)
    print(json.dumps({
        "status": "silver_v3_published_and_verified",
        "paths": result["paths"],
        "records": verified["records"],
        "relabeled_rows": result["report"]["relabeled_rows"],
        "class_counts": result["report"]["accepted_domain_counts"],
    }, ensure_ascii=False, indent=2))


def _write_new_json(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("analyze", "pilot", "prepare", "build"):
        command = subparsers.add_parser(name)
        command.add_argument("--data-lake-root", type=Path, required=True)
        command.add_argument("--silver-v2-report", required=True)
        if name == "analyze":
            command.add_argument("--output", type=Path)
            command.add_argument("--top-n", type=int, default=30)
            command.add_argument("--examples-per-domain", type=int, default=3)
        elif name == "pilot":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--sample-size", type=int, default=300)
            command.add_argument("--seed", type=int, default=42)
        elif name == "prepare":
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument("--decisions", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "analyze":
        analyze(args)
    elif args.command == "pilot":
        pilot(args)
    elif args.command == "prepare":
        prepare(args)
    else:
        build(args)


if __name__ == "__main__":
    main()
