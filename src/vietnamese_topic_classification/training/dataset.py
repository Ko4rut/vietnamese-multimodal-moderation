"""Gold-to-training adapter boundary.

Implementation will call GoldReader.verify() and iter_split() without
modifying the immutable Gold dataset or mixing train/validation/test data.
"""

from __future__ import annotations

from pathlib import Path

from .model import DatasetSplits


def load_gold_splits(*, data_lake_root: Path, manifest_name: str) -> DatasetSplits:
    """Read aligned texts, IDs and label_ids from a verified Gold snapshot."""
    raise NotImplementedError("Skeleton only: implement the verified Gold dataset adapter")
