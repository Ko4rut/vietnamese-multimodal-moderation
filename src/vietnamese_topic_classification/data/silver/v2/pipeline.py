"""Silver V1 -> immutable Silver V2, with conservative suffix cleaning."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from ..exceptions import SilverPublicationError
from ..reader import SilverReader
from ..v1.model import SilverRecord
from .cleaner import RULESET_VERSION, SilverV2Cleaner
from .corpus import build_corpus_stats
from .model import SilverV2AuditRow, SilverV2Report, SilverV2RunResult

VERSION = "v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(handle: TextIO, obj: dict[str, Any]) -> None:
    handle.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")


def _publish(src: Path, dest: Path) -> None:
    if dest.exists():
        raise SilverPublicationError(f"Refusing to overwrite Silver V2: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    pending = dest.with_name(dest.name + ".uploading")
    if pending.exists():
        raise SilverPublicationError(f"Stale upload: {pending}")
    try:
        shutil.copyfile(src, pending)
        if _sha256(src) != _sha256(pending) or src.stat().st_size != pending.stat().st_size:
            raise SilverPublicationError("Upload integrity verification failed")
        if dest.exists():
            raise SilverPublicationError(f"Destination appeared during upload: {dest}")
        pending.rename(dest)
    finally:
        pending.unlink(missing_ok=True)


class SilverV2Pipeline:
    """Read a pinned Silver V1 report; never mutate Silver V1 or Gold V1.

    Dry-run computes stats only, no writes to the Data Lake. Actual publishing
    writes V2 data, rejected duplicate-cleaned texts, audit rows, and report.
    """

    def __init__(
        self, reader: SilverReader, cleaner: SilverV2Cleaner | None = None,
        output_root: Path | None = None,
    ) -> None:
        self.reader = reader
        self.cleaner = cleaner or SilverV2Cleaner()
        self.root = Path(output_root) if output_root is not None else reader.root
        version = reader.report.get("pipeline_version")
        if version not in (None, "v1"):
            raise ValueError(f"Silver V2 requires a Silver V1 source, not {version!r}")

    def planned_paths(self, source_sha256: str) -> dict[str, Path]:
        params = {
            "source_sha256": source_sha256,
            "ruleset": RULESET_VERSION,
            "cleaner": asdict(self.cleaner.config),
        }
        config_hash = hashlib.sha256(
            json.dumps(params, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:10]
        suffix = f"{source_sha256[:12]}_{config_hash}"
        return {
            "silver": self.root / "02_silver" / VERSION
            / f"vtsnlp_silver_{VERSION}_{suffix}.jsonl",
            "rejected": self.root / "02_silver" / VERSION
            / f"vtsnlp_rejected_{VERSION}_{suffix}.jsonl",
            "audit": self.root / "05_reports" / "silver" / VERSION
            / f"silver_audit_{VERSION}_{suffix}.jsonl",
            "report": self.root / "05_reports" / "silver" / VERSION
            / f"silver_quality_{VERSION}_{suffix}.json",
        }

    def run(self, *, dry_run: bool = False) -> SilverV2RunResult:
        verified = self.reader.verify()
        paths = self.planned_paths(verified["sha256"])
        # Also respect pre-migration flat snapshots: publication is write-once.
        legacy_paths = [
            self.root / "02_silver" / paths["silver"].name,
            self.root / "02_silver" / paths["rejected"].name,
            self.root / "05_reports" / paths["audit"].name,
            self.root / "05_reports" / paths["report"].name,
        ]
        if not dry_run and any(
            path.exists() or path.with_name(path.name + ".uploading").exists()
            for path in (*paths.values(), *legacy_paths)
        ):
            raise SilverPublicationError("Silver V2 output exists or stale upload found")

        corpus = build_corpus_stats(self.reader.iter_records())
        if corpus.documents != verified["records"]:
            raise RuntimeError("Silver V2 corpus prepass row count mismatch")
        counts: Counter[str] = Counter()
        signals: Counter[str] = Counter()
        labels: Counter[str] = Counter()
        seen_texts: dict[str, str] = {}
        seen_ids: set[str] = set()

        with tempfile.TemporaryDirectory(prefix="vtc_silver_v2_") as tempdir:
            local = {name: Path(tempdir) / path.name for name, path in paths.items()}
            with (
                local["silver"].open("w", encoding="utf-8", newline="\n") as accepted,
                local["rejected"].open("w", encoding="utf-8", newline="\n") as rejected,
                local["audit"].open("w", encoding="utf-8", newline="\n") as audit,
            ):
                for lineno, original in enumerate(self.reader.iter_records(), 1):
                    counts["input"] += 1
                    # Validate even if source report was corrupt but matched checksum.
                    if (
                        not isinstance(original.get("id"), str)
                        or not original["id"].strip()
                        or not isinstance(original.get("text"), str)
                        or not original["text"].strip()
                        or not isinstance(original.get("domain"), str)
                        or not isinstance(original.get("quality_flags"), list)
                    ):
                        raise ValueError(f"Invalid pinned Silver V1 record at line {lineno}")
                    decision = self.cleaner.clean(original["text"], corpus=corpus)
                    if decision.action != "unchanged":
                        counts["detected"] += 1
                    counts[decision.action] += 1
                    for name in decision.signals:
                        signals[name] += 1
                    record: SilverRecord = {
                        "id": original["id"],
                        "text": decision.text,
                        "domain": original["domain"],
                        "text_length": len(decision.text),
                        "source_snapshot": original.get("source_snapshot", self.reader.data_path.name),
                        "quality_flags": list(original["quality_flags"]),
                    }
                    if decision.action != "unchanged":
                        flag = "silver_v2:" + decision.action
                        if flag not in record["quality_flags"]:
                            record["quality_flags"].append(flag)

                    # Cleaned texts may become identical. Reject rather than silently
                    # leaking exact duplicates into a future Gold snapshot.
                    rid = original["id"].strip()
                    digest = hashlib.sha256(decision.text.encode("utf-8")).hexdigest()
                    reason = None
                    if rid in seen_ids:
                        reason = "duplicate_id_after_cleaning"
                    elif digest in seen_texts:
                        reason = (
                            "duplicate_text_conflicting_domain_after_cleaning"
                            if seen_texts[digest] != record["domain"]
                            else "duplicate_text_after_cleaning"
                        )

                    if reason is not None:
                        counts["rejected"] += 1
                        counts["reject:" + reason] += 1
                        _jsonl(rejected, {
                            "id": rid,
                            "domain": original["domain"],
                            "source_line": lineno,
                            "source_snapshot": self.reader.data_path.name,
                            "reason": reason,
                        })
                    else:
                        _jsonl(accepted, record)
                        counts["accepted"] += 1
                        labels[record["domain"]] += 1
                        seen_ids.add(rid)
                        seen_texts[digest] = record["domain"]

                    if decision.action != "unchanged" or reason is not None:
                        audit_row: SilverV2AuditRow = {
                            "id": rid,
                            "domain": original["domain"],
                            "action": decision.action,
                            "signals": list(decision.signals),
                            "before_chars": len(original["text"]),
                            "after_chars": len(decision.text),
                            "removed_chars": decision.removed_chars,
                            "removed_preview": decision.removed_preview,
                        }
                        _jsonl(audit, {**audit_row, "rejection_reason": reason})
                    counts["removed_total_chars"] += decision.removed_chars

            if counts["input"] != verified["records"]:
                raise RuntimeError("Silver source changed during Silver V2 streaming")
            report: SilverV2Report = {
                "pipeline_version": VERSION,
                "ruleset_version": RULESET_VERSION,
                "corpus_stats": {
                    "documents": corpus.documents,
                    "min_document_frequency": corpus.min_document_frequency,
                    "repeated_ui_blocks": len(corpus.document_frequency),
                },
                "cleaning_config": asdict(self.cleaner.config),
                "source_report": self.reader.report_path.name,
                "source_silver_sha256": verified["sha256"],
                "source_rows": verified["records"],
                "input_rows": counts["input"],
                "accepted_rows": counts["accepted"],
                "rejected_rows": counts["rejected"],
                "rejected_reasons": dict(sorted(
                    (name.removeprefix("reject:"), value)
                    for name, value in counts.items() if name.startswith("reject:")
                )),
                "cleaned_rows": counts["cleaned"],
                "review_rows": counts["review"],
                "guardrail_rows": counts["guardrail"],
                "detected_rows": counts["detected"],
                "removed_total_chars": counts["removed_total_chars"],
                "signal_counts": dict(sorted(signals.items())),
                "accepted_domain_counts": dict(sorted(labels.items())),
                "output_files": {key: path.name for key, path in paths.items()},
                "silver_sha256": _sha256(local["silver"]),
                "rejected_sha256": _sha256(local["rejected"]),
                "audit_sha256": _sha256(local["audit"]),
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }
            local["report"].write_text(
                json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            if dry_run:
                return {"paths": {}, "report": report, "dry_run": True}

            published: list[Path] = []
            try:
                for key in ("silver", "rejected", "audit", "report"):
                    _publish(local[key], paths[key])
                    published.append(paths[key])
            except Exception:
                for p in reversed(published):
                    p.unlink(missing_ok=True)
                raise
            return {
                "paths": {key: str(path) for key, path in paths.items()},
                "report": report,
                "dry_run": False,
            }
