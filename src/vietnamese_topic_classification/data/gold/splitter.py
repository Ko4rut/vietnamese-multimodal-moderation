"""Deterministic, approximately stratified splitting with group leakage barriers.

The one-permutation MinHash buckets find candidate text pairs, then exact
word-shingle Jaccard is used to join matches. Candidate search is HEURISTIC:
undetected near duplicates may remain. Never describe it as an exhaustive
all-pairs audit. A candidate explosion fails closed rather than silently skipping.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from typing import Any

from .config import GoldConfig
from .exceptions import GoldLeakageError

SPLITS = ("train", "validation", "test")


def _shingles(text: str, width: int) -> set[int]:
    tokens = text.casefold().split()
    if not tokens:
        return set()
    if len(tokens) < width:
        tokens = [" ".join(tokens)]
        width = 1
    result: set[int] = set()
    for i in range(len(tokens) - width + 1):
        phrase = " ".join(tokens[i : i + width]).encode("utf-8")
        result.add(int.from_bytes(hashlib.blake2b(phrase, digest_size=8).digest(), "big"))
    return result


class GoldSplitter:
    def __init__(self, config: GoldConfig | None = None) -> None:
        self.config = config or GoldConfig()

    def groups(self, rows: list[dict[str, Any]]) -> tuple[list[int], dict[str, Any]]:
        """Union verified similar records; never split identified pairs across outputs."""
        size = len(rows)
        parent = list(range(size))
        ranks = [0] * size

        def find(value: int) -> int:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def union(a: int, b: int) -> None:
            x, y = find(a), find(b)
            if x == y:
                return
            if ranks[x] < ranks[y]:
                x, y = y, x
            parent[y] = x
            if ranks[x] == ranks[y]:
                ranks[x] += 1

        buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        checked = 0
        near_pairs = 0
        width = self.config.shingle_words
        count = self.config.signature_buckets
        for index, row in enumerate(rows):
            shingles = _shingles(row["text"], width)
            signature: dict[int, int] = {}
            for token in shingles:
                bucket = token % count
                value = token // count
                if bucket not in signature or value < signature[bucket]:
                    signature[bucket] = value
            hits: Counter[int] = Counter()
            for bucket, value in signature.items():
                for previous in buckets[(bucket, value)]:
                    hits[previous] += 1
            candidates = [
                previous for previous, hits_count in hits.items()
                if hits_count >= min(2, len(signature))
            ]
            if len(candidates) > self.config.max_candidates_per_record:
                raise GoldLeakageError(
                    f"Row {index}: {len(candidates)} candidate neighbors exceed "
                    f"max_candidates_per_record={self.config.max_candidates_per_record}. "
                    "Use a dedicated near-duplicate audit or adjust the safety limit."
                )
            for previous in sorted(candidates):
                other = _shingles(rows[previous]["text"], width)
                overlap = len(shingles & other)
                total = len(shingles) + len(other) - overlap
                if total:
                    checked += 1
                    if overlap / total >= self.config.similarity_threshold:
                        near_pairs += 1
                        union(index, previous)
            for bucket, value in signature.items():
                buckets[(bucket, value)].append(index)

        result = [find(i) for i in range(size)]
        return result, {
            "method": "one_permutation_minhash_candidate_search_plus_exact_word_jaccard",
            "exhaustive": False,
            "candidates_checked": checked,
            "matched_pairs": near_pairs,
            "groups": len(set(result)),
        }

    def split(self, rows: list[dict[str, Any]]) -> tuple[dict[str, list[int]], dict[str, Any]]:
        if not rows:
            raise ValueError("Cannot split an empty Silver snapshot")
        roots, audit = self.groups(rows)
        groups: dict[int, list[int]] = defaultdict(list)
        for index, root in enumerate(roots):
            groups[root].append(index)

        rng = random.Random(self.config.seed)
        bundles = list(groups.values())
        rng.shuffle(bundles)
        bundles.sort(key=len, reverse=True)

        total_labels = Counter(row["domain"] for row in rows)
        ratios = dict(zip(
            SPLITS,
            (self.config.train_ratio, self.config.validation_ratio, self.config.test_ratio),
        ))
        allocated: dict[str, list[int]] = {name: [] for name in SPLITS}
        allocated_labels: dict[str, Counter[str]] = {name: Counter() for name in SPLITS}
        for bundle in bundles:
            bundle_counts = Counter(rows[i]["domain"] for i in bundle)

            def score(name: str) -> float:
                by_class = sum(
                    n * allocated_labels[name][label] / max(0.01, total_labels[label] * ratios[name])
                    for label, n in bundle_counts.items()
                ) / len(bundle)
                global_ratio = len(allocated[name]) / max(0.01, len(rows) * ratios[name])
                return by_class + 0.1 * global_ratio

            chosen = min(SPLITS, key=score)
            allocated[chosen].extend(bundle)
            allocated_labels[chosen].update(bundle_counts)

        for name in SPLITS:
            allocated[name].sort()
        if sum(map(len, allocated.values())) != len(rows):
            raise AssertionError("Gold splitting lost records")
        assignments = {
            index: split for split, indices in allocated.items() for index in indices
        }
        for bundle in bundles:
            if len({assignments[index] for index in bundle}) != 1:
                raise AssertionError("Near-duplicate group leaked across splits")
        audit["split_rows"] = {name: len(allocated[name]) for name in SPLITS}
        audit["split_class_counts"] = {
            name: dict(sorted(allocated_labels[name].items())) for name in SPLITS
        }
        audit["target_ratios"] = ratios
        audit["actual_ratios"] = {
            name: len(allocated[name]) / len(rows) for name in SPLITS
        }
        audit["note"] = (
            "Approximate stratification preserves discovered groups; candidate search "
            "may miss near duplicates. Review deviations and label quality before training."
        )
        return allocated, audit
