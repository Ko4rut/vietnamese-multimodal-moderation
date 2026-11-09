"""Read-only access to Bronze/Silver/Gold snapshots in the Data Lake (existing readers)."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from vietnamese_topic_classification.data.bronze import BronzeReader

MAX_EXTRA_SNAPSHOTS = 6


@dataclass
class BronzeInfo:
    """Bronze 50k: its IDs (to exclude from the survey sample) and label counts."""

    manifest_name: str
    records: int
    sha256: str
    ids: set[str] = field(repr=False)
    label_counts: Counter = field(repr=False)


def find_bronze_manifest(lake_root: Path, name: str | None = None) -> str:
    """Pick the Bronze manifest: explicit name, else the single 50k/seed42 one."""
    directory = Path(lake_root) / "00_source_metadata"
    available = sorted(p.name for p in directory.glob("source_manifest_*.json"))
    if name is not None:
        if name not in available:
            raise FileNotFoundError(
                f"Không thấy Bronze manifest '{name}' trong {directory}. Có sẵn: {available}"
            )
        return name
    preferred = [n for n in available if "_seed42_" in n and n.endswith("_n50000.json")]
    chosen = preferred or available
    if not chosen:
        raise FileNotFoundError(
            f"Không có source_manifest_*.json trong {directory}. Chạy notebook Bronze trước, "
            "hoặc đặt USE_DRIVE=False để bỏ qua phần so sánh với Bronze 50k."
        )
    if len(chosen) > 1:
        print(f"Có {len(chosen)} Bronze manifest phù hợp, chọn bản đầu tiên: {chosen}")
    return chosen[0]


def load_bronze(lake_root: Path, manifest_name: str | None = None, *,
                log: Callable[[str], None] = print) -> BronzeInfo:
    name = find_bronze_manifest(lake_root, manifest_name)
    log(f"Bronze manifest được dùng: {name}")
    reader = BronzeReader(Path(lake_root), name)
    verified = reader.verify()
    ids: set[str] = set()
    counts: Counter = Counter()
    for record in reader.iter_records():
        row_id = record.get("id")
        if isinstance(row_id, str):
            ids.add(row_id)
        domain = record.get("domain")
        counts["<null>" if domain is None else str(domain)] += 1
    return BronzeInfo(name, verified["records"], verified["sha256"], ids, counts)


def _glob_names(directories: list[Path], pattern: str) -> list[tuple[Path, str]]:
    seen: dict[str, Path] = {}
    for directory in directories:
        for path in sorted(directory.glob(pattern)):
            seen.setdefault(path.name, directory)
    return [(d, n) for n, d in sorted(seen.items())]


def snapshot_label_counts(lake_root: Path, *, log: Callable[[str], None] = print
                          ) -> dict[str, dict[str, Any]]:
    """Label counts of Silver V1/V2 and Gold V1/V2 *train* (best effort, read-only).

    A snapshot that cannot be read is reported with its error instead of aborting.
    """
    from vietnamese_topic_classification.data.gold import GoldReader
    from vietnamese_topic_classification.data.silver import SilverReader

    root = Path(lake_root)
    results: dict[str, dict[str, Any]] = {}
    silver = _glob_names(
        [root / "04_reports" / "silver" / "v1", root / "04_reports" / "silver" / "v2",
         root / "04_reports"], "silver_quality_*.json")
    gold = _glob_names(
        [root / "03_gold" / "manifests", root / "03_gold" / "v2" / "manifests"],
        "gold_manifest_*.json")
    jobs = [("silver", n) for _d, n in silver] + [("gold", n) for _d, n in gold]
    if len(jobs) > MAX_EXTRA_SNAPSHOTS:
        log(f"Có {len(jobs)} snapshot Silver/Gold; chỉ đếm {MAX_EXTRA_SNAPSHOTS} cái đầu.")
        jobs = jobs[:MAX_EXTRA_SNAPSHOTS]
    for kind, name in jobs:
        key = re.sub(r"\.json$", "", name)
        try:
            if kind == "silver":
                reader: Any = SilverReader(root, name)
                reader.verify()
                records = reader.iter_records()
            else:
                reader = GoldReader(root, name)
                reader.verify()
                records = reader.iter_split("train")
            counts: Counter = Counter()
            for record in records:
                domain = record.get("domain")
                counts["<null>" if domain is None else str(domain)] += 1
            label = f"{kind}{' train' if kind == 'gold' else ''}: {key}"
            results[label] = {"status": "ok", "rows": sum(counts.values()), "counts": dict(counts)}
            log(f"  đã đếm {label}: {sum(counts.values()):,} dòng")
        except Exception as exc:  # reader errors are heterogeneous
            results[f"{kind}: {key}"] = {"status": "unavailable",
                                         "reason": f"{type(exc).__name__}: {exc}"}
            log(f"  bỏ qua {kind} {key}: {type(exc).__name__}: {exc}")
    return results
