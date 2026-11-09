"""Read the pinned VTSNLP parquet shards without downloading the ``text`` column.

Why not simply ``HfFileSystem.open``: every range request through it is resolved
again (Hub -> CDN redirect). Measured on the real dataset this costs ~0.5-1 s per
request, and reading only ``domain`` needs 93 tiny requests per file (132 files).
Here ``HfFileSystem`` is used for listing, and a small keep-alive HTTP range reader
(standard library only) is used for the data, so many tiny range requests can run
in parallel. A parquet file stores column chunks separately, hence reading only
``domain`` (~1 KB per row group) never transfers the article text.

Heavy imports (``pyarrow``, ``huggingface_hub``) are lazy so offline CI stays light.
"""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import math
import random
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable, Mapping, Protocol, Sequence

from .config import SurveyConfig

NULL_LABEL = "<null>"
EMPTY_LABEL = "<empty>"
SAMPLE_COLUMNS = ("id", "text", "domain")
USER_AGENT = "vietnamese-topic-classification-label-survey/1.0"
_TAIL_WINDOW = 512 * 1024
_MAX_REQUEST = 16 * 1024 * 1024


def _log(message: str) -> None:
    print(message, flush=True)


# --------------------------------------------------------------------------- sources


@dataclass(frozen=True)
class ParquetFileInfo:
    """One parquet shard: ``index`` is its position in the sorted shard list."""

    index: int
    name: str
    path: str
    size_bytes: int


class ParquetSource(Protocol):
    def list_files(self) -> list[ParquetFileInfo]: ...

    def open(self, info: ParquetFileInfo) -> BinaryIO: ...


class LocalParquetSource:
    """Parquet shards in a local directory (used by offline tests)."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def list_files(self) -> list[ParquetFileInfo]:
        paths = sorted(self.directory.glob("*.parquet"))
        return [
            ParquetFileInfo(i, p.name, str(p), p.stat().st_size) for i, p in enumerate(paths)
        ]

    def open(self, info: ParquetFileInfo) -> BinaryIO:
        return open(info.path, "rb")


class RangeTransferStats:
    """Thread-safe counters for what was actually requested over HTTP."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests = 0
        self.bytes = 0

    def add(self, size: int) -> None:
        with self._lock:
            self.requests += 1
            self.bytes += size

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {"requests": self.requests, "bytes": self.bytes}


_LOCAL = threading.local()


def _connection(scheme: str, netloc: str, timeout: float) -> http.client.HTTPConnection:
    cache = getattr(_LOCAL, "connections", None)
    if cache is None:
        cache = _LOCAL.connections = {}
    key = (scheme, netloc)
    conn = cache.get(key)
    if conn is None:
        factory = (
            http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
        )
        conn = cache[key] = factory(netloc, timeout=timeout)
    return conn


def _drop_connection(scheme: str, netloc: str) -> None:
    conn = getattr(_LOCAL, "connections", {}).pop((scheme, netloc), None)
    if conn is not None:
        conn.close()


class HttpRangeFetcher:
    """Fetch byte ranges of one URL; redirects are resolved once and then cached."""

    def __init__(
        self,
        url: str,
        *,
        stats: RangeTransferStats | None = None,
        timeout: float = 60.0,
        retries: int = 5,
        backoff: float = 0.5,
    ) -> None:
        self.url = url
        self.stats = stats or RangeTransferStats()
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self._final_url: str | None = None
        self._lock = threading.Lock()

    def _request(self, url: str, start: int, length: int) -> tuple[int, dict[str, str], bytes]:
        parts = urllib.parse.urlsplit(url)
        target = parts.path + (f"?{parts.query}" if parts.query else "")
        headers = {
            "Range": f"bytes={start}-{start + length - 1}",
            "User-Agent": USER_AGENT,
            "Accept-Encoding": "identity",
        }
        conn = _connection(parts.scheme, parts.netloc, self.timeout)
        try:
            conn.request("GET", target, headers=headers)
            response = conn.getresponse()
            body = response.read()
        except (OSError, http.client.HTTPException):
            _drop_connection(parts.scheme, parts.netloc)
            raise
        return response.status, {k.lower(): v for k, v in response.getheaders()}, body

    def _resolve(self) -> str:
        """Follow redirects manually (the CDN URL is signed and reusable for a while)."""
        url = self.url
        for _ in range(6):
            status, headers, _body = self._request(url, 0, 1)
            if status in (301, 302, 303, 307, 308) and headers.get("location"):
                url = urllib.parse.urljoin(url, headers["location"])
                continue
            if status == 206:
                return url
            raise OSError(f"HTTP {status} while resolving {self.url}")
        raise OSError(f"Too many redirects while resolving {self.url}")

    def fetch(self, start: int, length: int) -> bytes:
        if length <= 0:
            return b""
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                with self._lock:
                    if self._final_url is None:
                        self._final_url = self._resolve()
                    url = self._final_url
                status, _headers, body = self._request(url, start, length)
                if status == 206:
                    self.stats.add(len(body))
                    return body
                if status in (401, 403, 404, 410):  # expired signed URL: resolve again
                    with self._lock:
                        self._final_url = None
                last_error = OSError(f"HTTP {status}")
                if status == 200:
                    raise OSError("Server ignored the Range header; refusing a full download")
            except (OSError, http.client.HTTPException) as exc:
                last_error = exc
            time.sleep(min(8.0, self.backoff * 2**attempt))
        raise OSError(f"Range request failed after {self.retries} attempts: {last_error}")


class RangeFile(io.RawIOBase):
    """Seekable read-only file over an HTTP range fetcher (what pyarrow needs)."""

    def __init__(self, fetcher: HttpRangeFetcher, size: int) -> None:
        super().__init__()
        self._fetcher = fetcher
        self._size = size
        self._pos = 0
        self._tail: tuple[int, bytes] | None = None

    @property
    def size(self) -> int:
        return self._size

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def _read_at(self, start: int, length: int) -> bytes:
        # Parquet footers sit at the end: one larger request serves all footer reads.
        if start + length > self._size - _TAIL_WINDOW:
            if self._tail is None:
                window_start = max(0, self._size - _TAIL_WINDOW)
                self._tail = (
                    window_start,
                    self._fetcher.fetch(window_start, self._size - window_start),
                )
            window_start, data = self._tail
            if start >= window_start:
                return data[start - window_start : start - window_start + length]
        out = bytearray()
        while len(out) < length:
            chunk = min(_MAX_REQUEST, length - len(out))
            out += self._fetcher.fetch(start + len(out), chunk)
        return bytes(out)

    def readinto(self, buffer: Any) -> int:
        length = min(len(buffer), self._size - self._pos)
        if length <= 0:
            return 0
        data = self._read_at(self._pos, length)
        buffer[: len(data)] = data
        self._pos += len(data)
        return len(data)


class HfParquetSource:
    """Pinned-revision train shards of a Hugging Face dataset repository."""

    _DATA_DIR = "data"

    def __init__(
        self, dataset: str, revision: str, *, stats: RangeTransferStats | None = None
    ) -> None:
        self.dataset = dataset
        self.revision = revision
        self.stats = stats or RangeTransferStats()
        self._fetchers: dict[str, HttpRangeFetcher] = {}
        self._lock = threading.Lock()

    def list_files(self) -> list[ParquetFileInfo]:
        from huggingface_hub import HfFileSystem  # lazy: optional dependency

        fs = HfFileSystem()
        base = f"datasets/{self.dataset}@{self.revision}/{self._DATA_DIR}"
        try:
            entries = fs.ls(base, detail=True)
        except Exception as exc:  # network / repo errors differ by hub version
            raise RuntimeError(
                "Không liệt kê được các file parquet trên Hugging Face. Kiểm tra kết nối "
                f"Internet của Colab rồi chạy lại. Chi tiết: {type(exc).__name__}: {exc}"
            ) from exc
        shards = sorted(
            (e for e in entries if Path(e["name"]).name.startswith("train-")
             and e["name"].endswith(".parquet")),
            key=lambda e: e["name"],
        )
        if not shards:
            raise RuntimeError(f"Không thấy file train-*.parquet trong {base}")
        return [
            ParquetFileInfo(i, Path(e["name"]).name, f"{self._DATA_DIR}/{Path(e['name']).name}",
                            int(e["size"]))
            for i, e in enumerate(shards)
        ]

    def open(self, info: ParquetFileInfo) -> BinaryIO:
        from huggingface_hub import hf_hub_url  # lazy: optional dependency

        with self._lock:
            fetcher = self._fetchers.get(info.name)
            if fetcher is None:
                url = hf_hub_url(
                    self.dataset, info.path, repo_type="dataset", revision=self.revision
                )
                fetcher = self._fetchers[info.name] = HttpRangeFetcher(url, stats=self.stats)
        return RangeFile(fetcher, info.size_bytes)  # type: ignore[return-value]


# ------------------------------------------------------------------- file metadata


@dataclass(frozen=True)
class FileMeta:
    """Parquet footer facts the survey needs (JSON-serializable)."""

    index: int
    name: str
    size_bytes: int
    num_rows: int
    rg_rows: tuple[int, ...]
    rg_bytes: Mapping[str, tuple[int, ...]]  # column -> compressed bytes per row group

    @property
    def num_row_groups(self) -> int:
        return len(self.rg_rows)

    def sample_cost(self, row_group: int) -> int:
        """Compressed bytes needed to read id/text/domain of one row group."""
        return sum(self.rg_bytes[c][row_group] for c in SAMPLE_COLUMNS if c in self.rg_bytes)

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["rg_rows"] = list(self.rg_rows)
        data["rg_bytes"] = {k: list(v) for k, v in self.rg_bytes.items()}
        return data

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "FileMeta":
        return cls(
            index=int(data["index"]),
            name=str(data["name"]),
            size_bytes=int(data["size_bytes"]),
            num_rows=int(data["num_rows"]),
            rg_rows=tuple(int(v) for v in data["rg_rows"]),
            rg_bytes={k: tuple(int(x) for x in v) for k, v in data["rg_bytes"].items()},
        )


def read_footer(source: ParquetSource, info: ParquetFileInfo) -> Any:
    """Return the pyarrow ``FileMetaData`` of one shard (footer only)."""
    import pyarrow.parquet as pq  # lazy

    with source.open(info) as handle:
        return pq.read_metadata(handle)


def meta_from_footer(info: ParquetFileInfo, md: Any) -> FileMeta:
    columns = [md.schema.column(i).name for i in range(md.num_columns)]
    rg_rows, rg_bytes = [], {name: [] for name in columns}
    for g in range(md.num_row_groups):
        group = md.row_group(g)
        rg_rows.append(group.num_rows)
        for c in range(group.num_columns):
            rg_bytes[columns[c]].append(int(group.column(c).total_compressed_size))
    return FileMeta(
        index=info.index,
        name=info.name,
        size_bytes=info.size_bytes,
        num_rows=int(md.num_rows),
        rg_rows=tuple(rg_rows),
        rg_bytes={k: tuple(v) for k, v in rg_bytes.items()},
    )


# ---------------------------------------------------------- 5.1 label counting scan


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def label_key(value: Any) -> str:
    """Count key of a raw ``domain`` value; null and empty are made explicit."""
    if value is None:
        return NULL_LABEL
    text = str(value)
    return text if text.strip() else EMPTY_LABEL


def _count_one_file(
    source: ParquetSource, info: ParquetFileInfo, cache_file: Path
) -> dict[str, Any]:
    import pyarrow.parquet as pq  # lazy

    if cache_file.is_file():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if sum(cached["counts"].values()) == cached["meta"]["num_rows"]:
            cached["from_cache"] = True
            return cached
    started = time.perf_counter()
    with source.open(info) as handle:
        parquet = pq.ParquetFile(handle)
        meta = meta_from_footer(info, parquet.metadata)
        column = parquet.read(columns=["domain"]).column("domain")
    counts: dict[str, int] = {}
    for item in column.value_counts().to_pylist():
        key = label_key(item["values"])
        counts[key] = counts.get(key, 0) + int(item["counts"])
    if sum(counts.values()) != meta.num_rows:
        raise ValueError(f"{info.name}: label counts {sum(counts.values())} != {meta.num_rows}")
    result = {
        "meta": meta.to_json(),
        "counts": dict(sorted(counts.items())),
        "seconds": round(time.perf_counter() - started, 2),
    }
    _atomic_write_json(cache_file, result)
    result["from_cache"] = False
    return result


def scan_label_counts(
    source: ParquetSource,
    infos: Sequence[ParquetFileInfo],
    cache_dir: Path,
    *,
    workers: int = 24,
) -> list[dict[str, Any]]:
    """Count ``domain`` values per shard, reading only that column; resumable.

    Each finished shard is cached as JSON, so an interrupted scan continues where it
    stopped. Returns one record per shard in shard order: ``meta``, ``counts``.
    """
    directory = Path(cache_dir) / "label_counts"
    started = time.perf_counter()
    results: dict[int, dict[str, Any]] = {}
    errors: list[str] = []
    done = fresh = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(_count_one_file, source, info, directory / f"{info.name}.json"): info
            for info in infos
        }
        for future in as_completed(futures):
            info = futures[future]
            try:
                results[info.index] = future.result()
            except Exception as exc:
                errors.append(f"{info.name}: {type(exc).__name__}: {exc}")
                continue
            done += 1
            if not results[info.index].get("from_cache"):
                fresh += 1
                if fresh % 10 == 0 or done == len(infos):
                    elapsed = time.perf_counter() - started
                    _log(f"  đã đếm {done}/{len(infos)} file ({elapsed:,.0f}s)")
    if errors:
        raise RuntimeError(
            f"{len(errors)} file lỗi khi đếm nhãn (các file đã xong được cache, chạy lại để "
            "tiếp tục):\n" + "\n".join(errors[:5])
        )
    return [results[info.index] for info in infos]


# ----------------------------------------------------------- 5.3 random survey sample


@dataclass(frozen=True)
class SamplePlan:
    strategy: Mapping[str, Any]
    items: tuple[tuple[int, int, int], ...]  # (file index, row group, rows to take)


def plan_sample(metas: Sequence[FileMeta], cfg: SurveyConfig) -> SamplePlan:
    """Choose which row groups to read, from the real parquet metadata.

    Row groups of this dataset hold ~1,000 consecutive rows, so reading whole row groups
    would give ~300 clumps of neighbouring articles. Instead we read as many randomly
    chosen row groups as the download budget allows and keep a random subset of rows
    inside each, which spreads the sample over many positions of every shard.
    """
    groups = [(m.index, g) for m in metas for g in range(m.num_row_groups)]
    if not groups:
        raise ValueError("Dataset metadata contains no row groups")
    by_index = {m.index: m for m in metas}
    random.Random(cfg.seed).shuffle(groups)

    budget = int(cfg.max_download_gb * 1e9)
    spent, affordable = 0, 0
    for file_index, g in groups:
        cost = by_index[file_index].sample_cost(g)
        if spent + cost > budget:
            break
        spent += cost
        affordable += 1
    affordable = max(1, affordable)

    wanted = math.ceil(cfg.survey_target_rows * 1.03)  # margin for ID exclusion / dedup
    max_rows = max(by_index[f].rg_rows[g] for f, g in groups)
    sizes = [by_index[f].rg_rows[g] for f, g in groups[:affordable]]
    if cfg.rows_per_row_group:
        rows_per_group = cfg.rows_per_row_group
    else:  # smallest per-group quota whose total capacity reaches the wanted rows
        low, high = 1, max_rows
        while low < high:
            mid = (low + high) // 2
            if sum(min(mid, n) for n in sizes) >= wanted:
                high = mid
            else:
                low = mid + 1
        rows_per_group = low
    items_list: list[tuple[int, int, int]] = []
    gathered = 0
    for f, g in groups[:affordable]:  # row groups can be uneven (e.g. a short last one)
        take = min(rows_per_group, by_index[f].rg_rows[g])
        items_list.append((f, g, take))
        gathered += take
        if gathered >= wanted:
            break
    items = tuple(items_list)
    expected_rows = gathered
    strategy = {
        "method": "random_row_groups_with_row_subsample",
        "seed": cfg.seed,
        "row_groups_total": len(groups),
        "row_groups_affordable_in_budget": affordable,
        "row_groups_planned": len(items),
        "rows_per_row_group": rows_per_group,
        "files_touched": len({f for f, _g, _t in items}),
        "expected_rows_before_exclusion": expected_rows,
        "target_rows": cfg.survey_target_rows,
        "estimated_download_bytes": sum(by_index[f].sample_cost(g) for f, g, _t in items),
        "budget_bytes": budget,
        "note": (
            "Row groups are drawn uniformly at random over all shards; inside a row group "
            "a seeded random subset of rows is kept."
        ),
    }
    if expected_rows < cfg.survey_target_rows:
        strategy["warning"] = "Budget or dataset too small to reach survey_target_rows"
    return SamplePlan(strategy=strategy, items=items)


def _atomic_write_parquet(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def _rows_for(seed: int, file_index: int, group: int, available: int, take: int) -> list[int]:
    if take >= available:
        return list(range(available))
    digest = hashlib.sha256(f"{seed}:{file_index}:{group}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    return sorted(rng.sample(range(available), take))


def _read_group_rows(
    source: ParquetSource,
    info: ParquetFileInfo,
    md: Any,
    group: int,
    rows: list[int],
    cfg: SurveyConfig,
    normalize: Callable[[str], str],
) -> dict[str, list[Any]]:
    import pyarrow as pa  # lazy
    import pyarrow.parquet as pq

    with source.open(info) as handle:
        table = pq.ParquetFile(handle, metadata=md).read_row_group(group, columns=list(SAMPLE_COLUMNS))
    if len(rows) != table.num_rows:
        table = table.take(pa.array(rows, type=pa.int64()))
    columns = {name: table.column(name).to_pylist() for name in SAMPLE_COLUMNS}
    out: dict[str, list[Any]] = {
        k: [] for k in ("id", "domain", "text", "n_chars", "n_tokens", "text_sha",
                        "file_idx", "row_group")
    }
    skipped = 0
    keep = cfg.stored_text_chars
    for row_id, text, domain in zip(columns["id"], columns["text"], columns["domain"]):
        if not isinstance(row_id, str) or not row_id or not isinstance(text, str):
            skipped += 1
            continue
        normalized = normalize(text)
        if not normalized:
            skipped += 1
            continue
        out["id"].append(row_id)
        out["domain"].append(None if domain is None else str(domain))
        out["text"].append(normalized[:keep])
        out["n_chars"].append(len(normalized))
        out["n_tokens"].append(len(normalized.split()))
        out["text_sha"].append(hashlib.blake2b(normalized.encode("utf-8"), digest_size=12).hexdigest())
        out["file_idx"].append(info.index)
        out["row_group"].append(group)
    out["_skipped"] = [skipped]
    return out


def fetch_sample(
    source: ParquetSource,
    infos: Sequence[ParquetFileInfo],
    plan: SamplePlan,
    *,
    cache_dir: Path,
    cfg: SurveyConfig,
    exclude_ids: Iterable[str] = (),
    normalize: Callable[[str], str] | None = None,
    batch_row_groups: int = 48,
) -> tuple[Any, dict[str, Any]]:
    """Download the planned row groups (resumable), return ``(DataFrame, stats)``.

    Rows whose ``id`` is in ``exclude_ids`` (the Bronze 50k) are dropped; the result is
    capped at ``survey_target_rows``. Parts are cached, so a rerun only fetches what is
    missing. Texts are NFC-normalized (Silver V1 rules, no Silver V2 cleaning); the stored
    text is cut at ``cfg.stored_text_chars`` while lengths and the content hash are
    computed on the full normalized text.
    """
    import pandas as pd  # lazy

    if normalize is None:
        from vietnamese_topic_classification.data.silver import SilverTransformer

        normalize = SilverTransformer.normalize_text
    excluded = set(exclude_ids)
    info_by_index = {i.index: i for i in infos}
    cache_dir = Path(cache_dir)
    footers: dict[int, Any] = {}

    def ensure_footers(batch: Sequence[tuple[int, int, int]]) -> None:
        missing = sorted({f for f, _g, _t in batch} - set(footers))
        if not missing:
            return
        began = time.perf_counter()
        with ThreadPoolExecutor(max_workers=cfg.http_workers) as pool:
            for idx, md in zip(
                missing, pool.map(lambda i: read_footer(source, info_by_index[i]), missing)
            ):
                footers[idx] = md
        _log(f"  đọc footer {len(missing)} file: {time.perf_counter() - began:,.0f}s")

    batches = [plan.items[i : i + batch_row_groups] for i in range(0, len(plan.items), batch_row_groups)]
    started = time.perf_counter()
    totals = {"rows_read": 0, "excluded_bronze_ids": 0, "skipped_invalid_rows": 0}
    frames = []
    for number, batch in enumerate(batches):
        part = cache_dir / "sample_parts" / f"part_{number:05d}.parquet"
        stats_file = part.with_suffix(".json")
        if part.is_file() and stats_file.is_file():
            frame = pd.read_parquet(part)
            part_stats = json.loads(stats_file.read_text(encoding="utf-8"))
        else:
            ensure_footers(batch)

            def work(item: tuple[int, int, int]) -> dict[str, list[Any]]:
                file_index, group, take = item
                md = footers[file_index]
                available = md.row_group(group).num_rows
                rows = _rows_for(cfg.seed, file_index, group, available, take)
                return _read_group_rows(source, info_by_index[file_index], md, group, rows, cfg,
                                        normalize)

            with ThreadPoolExecutor(max_workers=cfg.http_workers) as pool:
                pieces = list(pool.map(work, batch))
            skipped = sum(p.pop("_skipped")[0] for p in pieces)
            frame = pd.DataFrame(
                {k: [v for p in pieces for v in p[k]] for k in pieces[0]}
            )
            read = len(frame)
            if excluded:
                mask = frame["id"].isin(excluded)
                frame = frame.loc[~mask].reset_index(drop=True)
            part_stats = {
                "rows_read": read + skipped,
                "excluded_bronze_ids": read - len(frame),
                "skipped_invalid_rows": skipped,
            }
            _atomic_write_json(stats_file, part_stats)
            _atomic_write_parquet(frame, part)
        for key in totals:
            totals[key] += part_stats[key]
        frames.append(frame)
        elapsed = time.perf_counter() - started
        _log(f"  mẫu: lô {number + 1}/{len(batches)} xong ({elapsed:,.0f}s)")

    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    frames.clear()  # free the per-batch copies before the final dedup/truncation
    before = len(frame)
    frame = frame.drop_duplicates(subset="id", keep="first").reset_index(drop=True)
    stats = dict(totals)
    stats["duplicate_ids_removed"] = before - len(frame)
    if len(frame) > cfg.survey_target_rows:
        frame = frame.iloc[: cfg.survey_target_rows].reset_index(drop=True)
    stats["rows_final"] = len(frame)
    stats["bronze_ids_given"] = len(excluded)
    stats["http"] = getattr(source, "stats", None) and source.stats.snapshot()  # type: ignore[attr-defined]
    return frame, stats
