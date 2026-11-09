"""Offline tests for reading parquet shards without the text column, and sampling."""
from __future__ import annotations

import http.server
import io
import json
import re
import threading
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("pandas")

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from vietnamese_topic_classification.analysis.label_survey import hf_parquet as hp  # noqa: E402
from vietnamese_topic_classification.analysis.label_survey import profiling  # noqa: E402
from vietnamese_topic_classification.analysis.label_survey.config import SurveyConfig  # noqa: E402


class CountingFile(io.RawIOBase):
    """Counts the bytes pyarrow really reads from a shard."""

    def __init__(self, path: str, counter: dict) -> None:
        super().__init__()
        self._handle = open(path, "rb")
        self._counter = counter

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._handle.seek(offset, whence)

    def tell(self) -> int:
        return self._handle.tell()

    def readinto(self, buffer) -> int:
        data = self._handle.read(len(buffer))
        buffer[: len(data)] = data
        self._counter["bytes"] += len(data)
        return len(data)

    def close(self) -> None:
        self._handle.close()
        super().close()


class CountingSource(hp.LocalParquetSource):
    def __init__(self, directory) -> None:
        super().__init__(directory)
        self.counter = {"bytes": 0}

    def open(self, info):
        return CountingFile(info.path, self.counter)


class NoOpenSource(hp.LocalParquetSource):
    def open(self, info):
        raise AssertionError("shard must not be opened when everything is cached")


def _big_text_shard(directory: Path) -> int:
    import random

    rng = random.Random(1)
    directory.mkdir(parents=True, exist_ok=True)
    texts = ["".join(rng.choices("abcdefghijklmnopqrstuvwxyz0123456789 ", k=20000)) for _ in range(300)]
    labels = (["Health"] * 150) + (["News"] * 140) + [None] * 10
    pq.write_table(
        pa.table({"text": texts, "id": [f"id{i}" for i in range(300)], "domain": labels}),
        directory / "train-00000-of-00001.parquet", row_group_size=100,
    )
    return (directory / "train-00000-of-00001.parquet").stat().st_size


def test_label_count_reads_only_the_domain_column(tmp_path: Path) -> None:
    size = _big_text_shard(tmp_path)
    source = CountingSource(tmp_path)
    scans = hp.scan_label_counts(source, source.list_files(), tmp_path / "cache", workers=2)
    assert scans[0]["counts"] == {"<null>": 10, "Health": 150, "News": 140}
    assert source.counter["bytes"] < 0.1 * size  # text (~900 KB) was never read


def test_scan_reports_unknown_null_labels_and_position(fake_shards) -> None:
    source = hp.LocalParquetSource(fake_shards["directory"])
    scans = hp.scan_label_counts(source, source.list_files(), fake_shards["directory"].parent / "c")
    counts = profiling.merge_counts([s["counts"] for s in scans])
    assert sum(counts.values()) == len(fake_shards["rows"])
    table = profiling.counts_table(counts, ["Sports", "News", "Science"])
    kinds = dict(zip(table["label"], table["kind"]))
    assert kinds["Adult"] == "unknown" and kinds["<null>"] == "null_value"
    assert table.set_index("label").loc["Science", "count"] == 0  # known label, never seen
    positions = profiling.file_position_table(scans, ["Sports", "News"])
    assert list(positions["file_index"]) == [0, 1, 2]
    drift = profiling.position_drift(positions, ["Sports", "News"])
    assert drift["pct_std"].max() > 1  # labels were made to drift with the file position


def test_scan_resumes_from_cache(fake_shards, tmp_path: Path) -> None:
    source = hp.LocalParquetSource(fake_shards["directory"])
    infos = source.list_files()
    first = hp.scan_label_counts(source, infos, tmp_path / "cache", workers=2)
    second = hp.scan_label_counts(NoOpenSource(fake_shards["directory"]), infos, tmp_path / "cache")
    assert [s["counts"] for s in first] == [s["counts"] for s in second]
    assert all(s["from_cache"] for s in second)


# ------------------------------------------------------------------ HTTP range reader


class _Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, files: dict[str, bytes], honour_range: bool = True) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.files = files
        self.generation = 1
        self.honour_range = honour_range
        self.served = 0
        self.resolves = 0

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:  # silence
        pass

    def do_GET(self) -> None:
        server: _Server = self.server  # type: ignore[assignment]
        if self.path.startswith("/resolve/"):
            server.resolves += 1
            name = self.path.rsplit("/", 1)[1]
            self.send_response(302)
            self.send_header("Location", f"/cdn/{name}?gen={server.generation}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        match = re.fullmatch(r"/cdn/([^?]+)\?gen=(\d+)", self.path)
        if not match or int(match.group(2)) != server.generation:
            self.send_response(403)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        data = server.files[match.group(1)]
        header = self.headers.get("Range", "")
        found = re.fullmatch(r"bytes=(\d+)-(\d+)", header)
        if not found or not server.honour_range:
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        start, end = int(found.group(1)), min(int(found.group(2)), len(data) - 1)
        body = data[start : end + 1]
        server.served += len(body)
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class HttpSource:
    def __init__(self, server: _Server, directory: Path, **fetcher_options) -> None:
        self.server = server
        self.paths = sorted(directory.glob("*.parquet"))
        self.options = fetcher_options
        self.stats = hp.RangeTransferStats()

    def list_files(self):
        return [hp.ParquetFileInfo(i, p.name, p.name, p.stat().st_size)
                for i, p in enumerate(self.paths)]

    def open(self, info):
        fetcher = hp.HttpRangeFetcher(f"{self.server.base}/resolve/{info.name}",
                                      stats=self.stats, **self.options)
        return hp.RangeFile(fetcher, info.size_bytes)


@pytest.fixture()
def http_server(tmp_path: Path):
    size = _big_text_shard(tmp_path / "shards")
    files = {p.name: p.read_bytes() for p in (tmp_path / "shards").glob("*.parquet")}
    server = _Server(files)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, tmp_path / "shards", size
    server.shutdown()
    server.server_close()


def test_http_range_reader_counts_labels_with_small_transfers(http_server, tmp_path) -> None:
    server, shards, size = http_server
    source = HttpSource(server, shards)
    scans = hp.scan_label_counts(source, source.list_files(), tmp_path / "cache", workers=2)
    assert scans[0]["counts"] == {"<null>": 10, "Health": 150, "News": 140}
    assert source.stats.bytes < 0.1 * size
    assert server.resolves >= 1


def test_http_range_reader_re_resolves_expired_signed_url(http_server) -> None:
    server, shards, _ = http_server
    source = HttpSource(server, shards, backoff=0.0)
    info = source.list_files()[0]
    with source.open(info) as handle:
        assert pq.read_metadata(handle).num_rows == 300
        server.generation += 1  # the signed CDN URL expires
        assert pq.ParquetFile(handle).read_row_group(0, columns=["domain"]).num_rows == 100
    assert server.resolves == 2


def test_http_range_reader_refuses_a_server_that_ignores_range(http_server) -> None:
    server, shards, _ = http_server
    server.honour_range = False
    source = HttpSource(server, shards, retries=2, backoff=0.0)
    with pytest.raises(OSError, match="Range"):
        source.open(source.list_files()[0]).read(10)


# --------------------------------------------------------------------------- sampling


def _metas(source, cache):
    scans = hp.scan_label_counts(source, source.list_files(), cache, workers=2)
    return [hp.FileMeta.from_json(s["meta"]) for s in scans]


def test_plan_sample_is_deterministic_and_respects_budget(fake_shards, tmp_path) -> None:
    source = hp.LocalParquetSource(fake_shards["directory"])
    metas = _metas(source, tmp_path / "c")
    cfg = SurveyConfig(survey_target_rows=600)
    plan = hp.plan_sample(metas, cfg)
    again = hp.plan_sample(metas, cfg)
    assert plan.items == again.items
    assert hp.plan_sample(metas, SurveyConfig(survey_target_rows=600, seed=7)).items != plan.items
    assert plan.strategy["expected_rows_before_exclusion"] >= 600
    assert len({(f, g) for f, g, _ in plan.items}) == len(plan.items)  # no row group twice
    # a tiny byte budget forces fewer row groups and a bigger share of rows per group
    total = sum(m.sample_cost(g) for m in metas for g in range(m.num_row_groups))
    small = hp.plan_sample(metas, SurveyConfig(survey_target_rows=600, max_download_gb=total / 3 / 1e9))
    assert small.strategy["estimated_download_bytes"] <= total / 3
    assert len(small.items) < len(plan.items)
    # impossible target is flagged instead of silently pretending
    huge = hp.plan_sample(metas, SurveyConfig(survey_target_rows=10**7))
    assert "warning" in huge.strategy


def test_fetch_sample_excludes_bronze_ids_and_resumes(fake_shards, tmp_path) -> None:
    directory = fake_shards["directory"]
    source = hp.LocalParquetSource(directory)
    infos = source.list_files()
    cfg = SurveyConfig(survey_target_rows=500, http_workers=2)
    plan = hp.plan_sample(_metas(source, tmp_path / "c"), cfg)
    first, _ = hp.fetch_sample(source, infos, plan, cache_dir=tmp_path / "s", cfg=cfg)
    bronze_ids = set(first["id"].iloc[:40])
    cache = tmp_path / "s2"
    frame, stats = hp.fetch_sample(source, infos, plan, cache_dir=cache, cfg=cfg,
                                   exclude_ids=bronze_ids)
    assert not bronze_ids & set(frame["id"])
    assert stats["excluded_bronze_ids"] == 40
    assert len(frame) <= 500 and frame["id"].is_unique
    # parts are cached: a second call needs no shard access but returns the same rows
    again, _ = hp.fetch_sample(
        NoOpenSource(directory), infos, plan, cache_dir=cache, cfg=cfg, exclude_ids=bronze_ids)
    # footers are still read, so only data access is cached; the rows must match exactly
    assert list(again["id"]) == list(frame["id"])


def test_fetch_sample_normalizes_and_stores_bounded_text(tmp_path) -> None:
    directory = tmp_path / "s"
    directory.mkdir()
    decomposed = "Tie\u0302\u0301ng  Vie\u0323\u0302t\r\n" + "x" * 6000
    pq.write_table(pa.table({"text": [decomposed, "", None, "ok"], "id": ["a", "b", "c", "d"],
                             "domain": ["News", "News", "News", None]}),
                   directory / "train-00000-of-00001.parquet")
    source = hp.LocalParquetSource(directory)
    cfg = SurveyConfig(survey_target_rows=10, http_workers=1)
    plan = hp.plan_sample(_metas(source, tmp_path / "c"), cfg)
    frame, stats = hp.fetch_sample(source, source.list_files(), plan,
                                   cache_dir=tmp_path / "cache", cfg=cfg)
    assert set(frame["id"]) == {"a", "d"} and stats["skipped_invalid_rows"] == 2
    row = frame.set_index("id").loc["a"]
    assert row["text"].startswith("Tiếng Việt\n")  # NFC + whitespace rules of Silver V1
    assert len(row["text"]) == cfg.stored_text_chars and row["n_chars"] > 6000
    assert frame.set_index("id").loc["d", "domain"] is None or frame.set_index("id").loc["d"].isna()["domain"]


def test_json_helpers_roundtrip() -> None:
    meta = hp.FileMeta(0, "x", 10, 3, (2, 1), {"text": (5, 4), "domain": (1, 1)})
    assert hp.FileMeta.from_json(json.loads(json.dumps(meta.to_json()))) == meta
    assert hp.label_key(None) == "<null>" and hp.label_key("  ") == "<empty>"
