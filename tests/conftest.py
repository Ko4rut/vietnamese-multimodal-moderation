"""Shared offline fixtures: tiny parquet shards that mimic the VTSNLP layout."""
from __future__ import annotations

import random
from pathlib import Path

import pytest

# (label, confusable partner or None). Partners share part of their vocabulary on purpose.
FAKE_LABELS = [
    ("Sports", "News"), ("News", "Sports"), ("Health", "Beauty_and_Fitness"),
    ("Beauty_and_Fitness", "Health"), ("Finance", None), ("Games", None),
]
FAKE_CANONICAL = [name for name, _ in FAKE_LABELS] + ["Science"]


def make_fake_shards(directory: Path, *, n_files: int = 3, groups: int = 4, rows: int = 150,
                     seed: int = 0) -> dict:
    """Write ``n_files`` parquet shards; the label mix drifts with the file position."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = random.Random(seed)
    vocab = {name: [f"{name[:4].lower()}{i}tiếng" for i in range(25)] for name, _ in FAKE_LABELS}
    shared = [f"chung{i}" for i in range(40)]
    directory.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for f in range(n_files):
        ids, texts, domains = [], [], []
        for g in range(groups):
            for r in range(rows):
                # file position shifts the label mix: file 0 favours the first labels
                weights = [3 if (i + f) % 3 == 0 else 1 for i in range(len(FAKE_LABELS))]
                name, partner = rng.choices(FAKE_LABELS, weights=weights)[0]
                words = rng.choices(vocab[name], k=28) + rng.choices(shared, k=8)
                if partner:
                    words += rng.choices(vocab[partner], k=10)
                rng.shuffle(words)
                row_id = f"doc-{f}-{g}-{r}"
                label = name
                if r == 7 and g == 0:
                    label = None  # null label
                if r == 8 and g == 1 and f == 0:
                    label = "Adult"  # label that is not in domains_v1
                ids.append(row_id)
                texts.append("  " + " ".join(words) + "\r\n")
                domains.append(label)
        # one exact duplicate text with a different label inside the file
        ids += [f"dup-{f}-a", f"dup-{f}-b"]
        texts += ["Bài trùng nội dung hoàn toàn."] * 2
        domains += ["Sports", "Finance"]
        table = pa.table({"text": texts, "id": ids, "domain": domains})
        pq.write_table(table, directory / f"train-{f:05d}-of-{n_files:05d}.parquet",
                       row_group_size=rows + (2 if False else 0))
        all_rows += list(zip(ids, domains))
    return {"directory": directory, "rows": all_rows, "n_files": n_files}


@pytest.fixture()
def fake_shards(tmp_path: Path) -> dict:
    return make_fake_shards(tmp_path / "shards")
