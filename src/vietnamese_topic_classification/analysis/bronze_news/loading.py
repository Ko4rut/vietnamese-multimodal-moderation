"""Read Bronze (read-only) and pick the News rows plus a bounded neighbor-label sample."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from vietnamese_topic_classification.analysis.label_survey.lake_inputs import find_bronze_manifest
from vietnamese_topic_classification.data.bronze import BronzeReader

from .config import BronzeNewsConfig


@dataclass
class BronzeNewsData:
    manifest_name: str
    sha256: str
    total_records: int
    label_counts: dict[str, int]
    news: pd.DataFrame = field(repr=False)
    neighbors: pd.DataFrame = field(repr=False)


def _text_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def load_bronze_news(
    lake_root: str | Path,
    cfg: BronzeNewsConfig,
    manifest_name: str | None = None,
    *,
    log: Callable[[str], None] = print,
) -> BronzeNewsData:
    """Verify the Bronze snapshot, then stream it once.

    All ``News`` rows are kept (columns ``id``/``text``; ``text`` is ``None`` when it is
    missing or not a string). Neighbor labels are reservoir-sampled (seeded, at most
    ``neighbor_per_label`` non-empty rows each) so memory stays bounded.
    """
    name = find_bronze_manifest(Path(lake_root), manifest_name)
    log(f"Bronze manifest được dùng: {name}")
    reader = BronzeReader(Path(lake_root), name)
    verified = reader.verify()  # size + sha256 + record count
    rng = random.Random(cfg.seed)
    wanted = set(cfg.neighbor_labels) if cfg.include_neighbor_labels else set()
    counts: dict[str, int] = {}
    news_rows: list[dict[str, object]] = []
    reservoirs: dict[str, list[dict[str, object]]] = {label: [] for label in wanted}
    seen: dict[str, int] = {label: 0 for label in wanted}
    for record in reader.iter_records():
        domain = record.get("domain")
        key = "<null>" if domain is None else str(domain)
        counts[key] = counts.get(key, 0) + 1
        text = _text_or_none(record.get("text"))
        if domain == cfg.news_label:
            news_rows.append({"id": record.get("id"), "text": text})
        elif domain in wanted and text is not None and text.strip():
            seen[domain] += 1
            row = {"id": record.get("id"), "text": text, "domain": domain}
            bucket = reservoirs[domain]
            if len(bucket) < cfg.neighbor_per_label:
                bucket.append(row)
            else:
                slot = rng.randrange(seen[domain])
                if slot < cfg.neighbor_per_label:
                    bucket[slot] = row
    news = pd.DataFrame(news_rows, columns=["id", "text"])
    flat = [row for label in sorted(reservoirs) for row in reservoirs[label]]
    neighbors = pd.DataFrame(flat, columns=["id", "text", "domain"])
    return BronzeNewsData(
        manifest_name=name,
        sha256=verified["sha256"],
        total_records=int(verified["records"]),
        label_counts=dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        news=news,
        neighbors=neighbors,
    )
