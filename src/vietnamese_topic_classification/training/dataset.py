"""Verified Gold -> modeling adapter.

The training entrypoint only materializes train and validation. The test split
is verified via checksum but is not loaded into the training estimator.
"""

from __future__ import annotations

from pathlib import Path

from vietnamese_topic_classification.data.gold import GoldReader

from .model import DatasetSplits, TextSplit


def _read_split(reader: GoldReader, split: str) -> TextSplit:
    result: TextSplit = {"ids": [], "texts": [], "labels": []}
    seen: set[str] = set()
    for row in reader.iter_split(split):
        record_id, text, label = row.get("id"), row.get("text"), row.get("label_id")
        if not isinstance(record_id, str) or not record_id.strip() or record_id in seen:
            raise ValueError(f"Invalid or duplicate Gold ID in {split}: {record_id!r}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Invalid Gold text for ID {record_id!r}")
        if type(label) is not int or label < 0 or label >= len(reader.manifest["label_to_id"]):
            raise ValueError(f"Invalid Gold label_id for ID {record_id!r}")
        seen.add(record_id)
        result["ids"].append(record_id)
        result["texts"].append(text)
        result["labels"].append(label)
    if not result["ids"]:
        raise ValueError(f"Gold split {split!r} must be nonempty for modeling")
    return result


def _reader(*, data_lake_root: Path, manifest_name: str) -> GoldReader:
    reader = GoldReader(data_lake_root, manifest_name)
    reader.verify()
    return reader


def load_gold_training_splits(*, data_lake_root: Path, manifest_name: str) -> tuple[TextSplit, TextSplit]:
    """Load only train and validation after verifying the entire Gold manifest."""
    reader = _reader(data_lake_root=data_lake_root, manifest_name=manifest_name)
    train = _read_split(reader, "train")
    validation = _read_split(reader, "validation")
    if set(train["ids"]) & set(validation["ids"]):
        raise ValueError("Train/validation ID leakage in Gold")
    return train, validation


def load_gold_splits(*, data_lake_root: Path, manifest_name: str) -> DatasetSplits:
    """Explicitly load all three splits, intended for controlled final evaluation."""
    reader = _reader(data_lake_root=data_lake_root, manifest_name=manifest_name)
    splits: DatasetSplits = {
        "train": _read_split(reader, "train"),
        "validation": _read_split(reader, "validation"),
        "test": _read_split(reader, "test"),
    }
    id_sets = [set(splits[name]["ids"]) for name in ("train", "validation", "test")]
    if any(id_sets[i] & id_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Gold split ID leakage")
    return splits
