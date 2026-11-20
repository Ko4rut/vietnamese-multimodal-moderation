"""Output location, no-overwrite protection, text guard and zip. Never writes into the lake."""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from vietnamese_topic_classification.analysis.label_survey.lake_inputs import is_writable_dir

COMPLETE_MARKER = "_COMPLETE.json"
OUTPUT_SUBDIR = "bronze_news_eda"
MY_DRIVE_FOLDER = "vmm_outputs"
INTERNAL_NAME = "cluster_excerpts_INTERNAL.csv"


def _inside(path: Path, parent: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def choose_output_root(data_lake_root: Path | None, my_drive: Path | None, local_fallback: Path,
                       *, override: str | Path | None = None) -> tuple[Path, str]:
    """Pick where results go. The Data Lake is NEVER a candidate (it may be Viewer-only and the
    rule is that this analysis never writes to it): override -> My Drive -> local disk."""
    if override:
        root = Path(override).expanduser()
        if data_lake_root is not None and _inside(root, Path(data_lake_root)):
            raise ValueError(f"OUTPUT_ROOT_OVERRIDE nằm trong Data Lake, không được ghi: {root}")
        if not is_writable_dir(root):
            raise OSError(f"Không ghi được vào OUTPUT_ROOT_OVERRIDE: {root}")
        return root, "OUTPUT_ROOT_OVERRIDE"
    if my_drive is not None:
        root = Path(my_drive) / MY_DRIVE_FOLDER
        if is_writable_dir(root):
            return root, "My Drive của bạn (Data Lake chỉ được đọc, không ghi vào đó)"
    return Path(local_fallback), "ổ đĩa tạm của Colab /content (mất khi ngắt kết nối)"


def run_dir(output_root: Path, run_id: str) -> Path:
    return Path(output_root) / OUTPUT_SUBDIR / run_id


def is_complete(directory: Path) -> bool:
    return (Path(directory) / COMPLETE_MARKER).is_file()


class TextGuard:
    """Refuses to write summaries that contain article text.

    A string value of >= ``min_chars`` characters that occurs verbatim in any article, or any
    string longer than ``max_chars``, is rejected.
    """

    def __init__(self, texts: Any, min_chars: int = 30, max_chars: int = 1000) -> None:
        self.haystack = "\n".join(t for t in texts if isinstance(t, str))
        self.min_chars, self.max_chars = min_chars, max_chars

    def _bad(self, value: str) -> bool:
        return len(value) > self.max_chars or (
            len(value) >= self.min_chars and value in self.haystack)

    def check(self, value: Any, where: str = "") -> None:
        if isinstance(value, str):
            if self._bad(value):
                raise ValueError(f"Nghi là văn bản bài viết trong {where}: {value[:40]!r}…")
        elif isinstance(value, dict):
            for key, item in value.items():
                self.check(item, f"{where}.{key}")
        elif isinstance(value, (list, tuple)):
            for item in value:
                self.check(item, where)

    def check_frame(self, frame: pd.DataFrame, where: str) -> None:
        for column in frame.select_dtypes(include=["object", "string"]).columns:
            for value in frame[column]:
                self.check(value, f"{where}[{column}]")


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    return str(value)


def _clean(value: Any) -> Any:
    """NaN/inf -> None so the files are strict JSON."""
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    if hasattr(value, "item") and not isinstance(value, (list, dict, str)):
        try:
            return _clean(value.item())
        except (ValueError, AttributeError):
            return value
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_clean(v) for v in value]
    return value


def write_json(path: Path, data: Any, *, guard: TextGuard | None = None) -> Path:
    if guard:
        guard.check(data, Path(path).name)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_clean(data), ensure_ascii=False, indent=2,
                              default=_json_default, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def write_csv(path: Path, frame: pd.DataFrame, *, guard: TextGuard | None = None) -> Path:
    if guard:
        guard.check_frame(frame, Path(path).name)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(path)
    return path


def mark_complete(directory: Path, run_id: str) -> None:
    write_json(Path(directory) / COMPLETE_MARKER, {
        "run_id": run_id, "completed_utc": datetime.now(timezone.utc).isoformat()})


def make_zip(directory: Path, zip_path: Path) -> Path:
    directory, zip_path = Path(directory), Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = zip_path.with_name(zip_path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file() and not path.name.endswith(".tmp"):
                archive.write(path, Path(directory.name) / path.relative_to(directory))
    tmp.replace(zip_path)
    return zip_path


def try_download(zip_path: Path) -> bool:
    """Ask Colab's browser to download the zip; never raises."""
    try:
        from google.colab import files  # type: ignore[import-not-found]

        files.download(str(zip_path))
        return True
    except Exception as exc:
        print(f"Không tự tải được file zip ({type(exc).__name__}). File nằm ở: {zip_path}\n"
              "Tải tay: mở bảng Files bên trái → tìm file → chuột phải → Download.")
        return False


def export_results(directory: Path, run_id: str, *, tables: dict[str, pd.DataFrame],
                   jsons: dict[str, Any], figures: Any, internal: pd.DataFrame,
                   report_md: str, guard: TextGuard) -> list[str]:
    """Write every deliverable into ``directory`` and ``_COMPLETE.json`` last.

    Refuses to touch a directory that is already complete. Only ``cluster_excerpts_INTERNAL.csv``
    may contain article text (internal use only); everything else goes through ``guard``.
    """
    directory = Path(directory)
    if is_complete(directory):
        raise FileExistsError(f"{directory.name} đã hoàn tất, không ghi đè")
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        write_csv(directory / name, frame, guard=guard)
    for name, data in jsons.items():
        write_json(directory / name, data, guard=guard)
    guard.check(report_md.splitlines(), "report.md")
    (directory / "report.md").write_text(report_md, encoding="utf-8")
    write_csv(directory / INTERNAL_NAME, internal)  # text on purpose: internal use only
    figures.write(directory / "figures")
    mark_complete(directory, run_id)
    return sorted(str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file())
