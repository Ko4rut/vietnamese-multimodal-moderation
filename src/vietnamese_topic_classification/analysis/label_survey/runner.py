"""Step-by-step orchestration used by the Colab notebook (and by offline tests).

Each ``step_*`` method is idempotent: finished steps leave a marker in
``<output>/steps`` and are skipped on rerun; heavy intermediates (per-file label counts,
the downloaded sample, out-of-fold probabilities) are cached outside the output folder,
so the output folder only holds deliverables. A folder with ``_COMPLETE.json`` is never
rewritten; ``finalize`` then only re-zips it.
"""

from __future__ import annotations

import gc
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from vietnamese_topic_classification.config import get_domain_labels

from . import audit, confusion, noise, profiling, report, scenarios
from .config import SurveyConfig
from .hf_parquet import (
    NULL_LABEL,
    HfParquetSource,
    ParquetSource,
    fetch_sample,
    plan_sample,
    scan_label_counts,
)
from .lake_inputs import BronzeInfo

LIMITATIONS = [
    "Nhãn `domain` do máy gán (nvidia/domain-classifier), chưa có người kiểm duyệt; mọi 'nghi gán sai' chỉ là nghi vấn.",
    "Mô hình phân tích (TF-IDF + Logistic Regression, out-of-fold) là công cụ đo, không phải mô hình chính thức; không so trực tiếp với baseline Gold.",
    "Mỗi nhãn được cắt tối đa `per_class_cap` bài khi train mô hình phân tích, nên các tỉ lệ bài trong tập này không phản ánh tỉ lệ ngoài đời; phần trăm 'toàn dataset' lấy từ mục 2.",
    "TF-IDF được fit một lần trên toàn bộ tập phân tích (không dùng nhãn) để tiết kiệm thời gian; bộ phân loại thì luôn dự đoán bài chưa từng thấy.",
    "Chuẩn hóa văn bản chỉ ở mức Silver V1 (NFC, khoảng trắng); KHÔNG áp dụng làm sạch Silver V2 (quảng cáo, điều hướng...), nên nhiễu crawl có thể làm tăng độ lẫn.",
    "Văn bản dài bị cắt (`text_char_limit` ký tự) khi đưa vào TF-IDF; bộ gán nhãn gốc chỉ đọc ~512 token đầu.",
    "Mẫu khảo sát là mẫu ngẫu nhiên các row group rồi lấy ngẫu nhiên dòng trong mỗi row group; nếu các bài gần nhau trong một row group giống nhau thì kích thước mẫu hiệu dụng nhỏ hơn số dòng.",
    "Số liệu kịch bản gộp/bỏ nhãn tính lại từ dự đoán cũ (không train lại) nên chỉ là ước lượng nhanh.",
    "Dataset chưa rõ license; không đưa văn bản bài viết vào báo cáo hoặc Git.",
]


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit(repo_dir: Path | None) -> str | None:
    if repo_dir is None:
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class SurveyRun:
    def __init__(
        self,
        cfg: SurveyConfig,
        output_root: str | Path,
        *,
        source: ParquetSource | None = None,
        bronze: BronzeInfo | None = None,
        snapshots: Mapping[str, Mapping[str, Any]] | None = None,
        bronze_skip_reason: str = "không dùng Data Lake (USE_DRIVE=False) hoặc không tìm thấy Bronze",
        canonical_labels: Sequence[str] | None = None,
        repo_dir: str | Path | None = None,
        log: Callable[[str], None] = print,
    ) -> None:
        self.cfg = cfg
        self.source = source or HfParquetSource(cfg.hf_dataset, cfg.hf_revision)
        self.bronze = bronze
        self.snapshots = dict(snapshots or {})
        self.bronze_skip_reason = bronze_skip_reason
        self.canonical = tuple(canonical_labels or get_domain_labels())
        self.repo_dir = Path(repo_dir) if repo_dir else None
        self.log = log
        extra = {"bronze_sha256": bronze.sha256 if bronze else None}
        self.run_id = cfg.run_id(extra)
        self.output_root = Path(output_root)
        self.out_dir = self.output_root / self.run_id
        cache_root = self.output_root / "_cache"
        self.counts_cache = cache_root / f"counts_{cfg.hf_revision[:10]}"
        self.sample_cache = cache_root / f"sample_{cfg.sample_hash(extra)[:12]}"
        self.model_cache = cache_root / f"model_{cfg.config_hash(extra)[:12]}"
        self.zip_path = self.output_root / f"label_survey_{self.run_id}.zip"
        self.complete = report.is_complete(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.summary: dict[str, Any] = {}
        self.timings: dict[str, float] = {}
        self._scans: list[dict[str, Any]] | None = None
        self._sample: pd.DataFrame | None = None
        self._model: dict[str, Any] | None = None

    # ------------------------------------------------------------- infrastructure

    def _step(self, name: str, estimate: str, work: Callable[[], Mapping[str, Any]]) -> bool:
        """Run ``work`` once; returns True if it was skipped because it was already done."""
        marker = self.out_dir / "steps" / f"{name}.json"
        if marker.is_file():
            data = json.loads(marker.read_text(encoding="utf-8"))
            self.summary.update(data["summary"])
            self.timings[name] = data["seconds"]
            self.log(f"✔ [{name}] đã có kết quả từ lần chạy trước, bỏ qua "
                     f"(lần đó mất {data['seconds']:,.0f}s).")
            return True
        if self.complete:
            raise RuntimeError(f"Thư mục {self.out_dir} đã hoàn tất nhưng thiếu bước {name}.")
        self.log(f"▶ [{name}] ước tính: {estimate}")
        started = time.perf_counter()
        part = dict(work())
        seconds = round(time.perf_counter() - started, 1)
        self.summary.update(part)
        self.timings[name] = seconds
        report.write_json(marker, {"summary": part, "seconds": seconds})
        self.log(f"✔ [{name}] xong sau {seconds:,.0f}s.")
        return False

    def table(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.out_dir / f"{name}.csv")

    def figure_path(self, name: str) -> Path:
        return self.out_dir / name

    def show_figures(self, *names: str) -> None:
        try:
            from IPython.display import Image, display
        except ImportError:  # plain Python run
            return
        for name in names:
            path = self.figure_path(name)
            if path.is_file():
                display(Image(filename=str(path)))

    def _save_csv(self, frame: pd.DataFrame, name: str) -> Path:
        path = self.out_dir / f"{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8")
        return path

    @property
    def known_labels(self) -> list[str]:
        return list(self.canonical)

    # ------------------------------------------------------------ 5.1 full counts

    def _load_scans(self) -> list[dict[str, Any]]:
        if self._scans is None:
            infos = self.source.list_files()
            self._scans = scan_label_counts(
                self.source, infos, self.counts_cache, workers=self.cfg.http_workers
            )
        return self._scans

    def step_full_counts(self) -> None:
        def work() -> dict[str, Any]:
            scans = self._load_scans()
            metas = [s["meta"] for s in scans]
            meta_table = pd.DataFrame(
                {
                    "file_index": [m["index"] for m in metas],
                    "file": [m["name"] for m in metas],
                    "size_mb": [round(m["size_bytes"] / 1e6, 1) for m in metas],
                    "rows": [m["num_rows"] for m in metas],
                    "row_groups": [len(m["rg_rows"]) for m in metas],
                    "rows_per_row_group_max": [max(m["rg_rows"]) for m in metas],
                    "text_compressed_mb": [round(sum(m["rg_bytes"]["text"]) / 1e6, 1) for m in metas],
                    "domain_compressed_kb": [round(sum(m["rg_bytes"]["domain"]) / 1e3, 1) for m in metas],
                }
            )
            self._save_csv(meta_table, "dataset_metadata")
            self.log("Metadata thật của dataset (đọc từ footer parquet):")
            self.log(meta_table.head(3).to_string(index=False))
            self.log("  ...")
            self.log(meta_table.tail(2).to_string(index=False))
            counts = profiling.merge_counts([s["counts"] for s in scans])
            table = profiling.counts_table(counts, self.known_labels)
            self._save_csv(table, "full_label_counts")
            known = [x for x in self.known_labels]
            extra = [x for x in table["label"] if x not in known and counts.get(x, 0) > 0]
            labels = known + extra
            positions = profiling.file_position_table(scans, labels)
            self._save_csv(positions, "file_position_label_pct")
            self._save_csv(profiling.position_drift(positions, labels), "label_position_drift")
            profiling.fig_full_label_distribution(table, self.figure_path("fig_full_label_distribution.png"))
            profiling.fig_label_by_file_position(positions, labels, self.figure_path("fig_label_by_file_position.png"))
            real = table[(table["kind"] == "known") & (table["count"] > 0)]
            http = getattr(self.source, "stats", None)
            return {
                "full_dataset": {
                    "n_files": len(metas),
                    "n_row_groups": int(meta_table["row_groups"].sum()),
                    "total_rows": int(meta_table["rows"].sum()),
                    "rows_per_row_group_max": int(meta_table["rows_per_row_group_max"].max()),
                    "text_compressed_gb": round(float(meta_table["text_compressed_mb"].sum()) / 1e3, 2),
                    "n_label_values": int((table["count"] > 0).sum()),
                    "unexpected_label_values": [
                        f"{r.label} ({r.count:,})" for r in table.itertuples()
                        if r.kind != "known" and r.count > 0
                    ],
                    "adult_count": int(counts.get("Adult", 0)),
                    "null_count": int(counts.get(NULL_LABEL, 0)),
                    "max_min_ratio": profiling.max_min_ratio(table),
                    "largest_label": str(real.iloc[0]["label"]) if len(real) else None,
                    "smallest_label": str(real.iloc[-1]["label"]) if len(real) else None,
                    "http_for_label_scan": http.snapshot() if http else None,
                }
            }

        self._step("full_counts", "vài phút lần đầu (đọc ~1 KB mỗi row group, không tải văn bản); "
                   "lần sau dùng cache vài giây", work)

    # ------------------------------------------------------- 5.2 Bronze vs full

    def _full_counts(self) -> dict[str, int]:
        table = self.table("full_label_counts")
        return {str(r.label): int(r.count) for r in table.itertuples()}

    def step_bronze_compare(self) -> None:
        def work() -> dict[str, Any]:
            if self.bronze is None:
                return {"bronze_vs_full": {"status": "skipped", "reason": self.bronze_skip_reason}}
            full = self._full_counts()
            labels = self._compare_labels(full, self.bronze.label_counts)
            table, stats = profiling.compare_distributions(full, self.bronze.label_counts, labels)
            self._save_csv(table, "bronze_vs_full")
            profiling.fig_paired_distribution(
                table, self.figure_path("fig_bronze_vs_full.png"), reference_name="Toàn bộ dataset",
                other_name="Bronze 50k", title="Mẫu Bronze 50k so với toàn bộ dataset (% bài)")
            rows = []
            for name, info in self.snapshots.items():
                if info.get("status") != "ok":
                    rows.append({"snapshot": name, "rows": None, "total_variation_distance": None,
                                 "max_abs_gap_pct_points": None, "note": info.get("reason", "")})
                    continue
                _t, st = profiling.compare_distributions(full, info["counts"],
                                                         self._compare_labels(full, info["counts"]))
                rows.append({"snapshot": name, "rows": info["rows"],
                             "total_variation_distance": st["total_variation_distance"],
                             "max_abs_gap_pct_points": st["max_abs_gap_pct_points"], "note": ""})
            if rows:
                self._save_csv(pd.DataFrame(rows), "snapshot_vs_full")
            return {"bronze_vs_full": {"status": "ok", "bronze_manifest": self.bronze.manifest_name,
                                       "records": self.bronze.records, **stats}}

        self._step("bronze_compare", "vài giây", work)

    def _compare_labels(self, full: Mapping[str, int], other: Mapping[str, int]) -> list[str]:
        extra = sorted((set(full) | set(other)) - set(self.known_labels))
        return [x for x in [*self.known_labels, *extra] if full.get(x, 0) or other.get(x, 0)]

    # ------------------------------------------------------------ 5.3 the sample

    def _release_sample(self) -> None:
        """Drop the in-memory sample (it is reloaded from its parquet cache on demand)."""
        self._sample = None
        gc.collect()

    @property
    def sample(self) -> pd.DataFrame:
        if self._sample is None:
            self._sample = pd.read_parquet(self.sample_cache / "survey_sample.parquet")
        return self._sample

    def step_sample(self) -> None:
        def work() -> dict[str, Any]:
            final = self.sample_cache / "survey_sample.parquet"
            stats_file = self.sample_cache / "sample_stats.json"
            if final.is_file() and stats_file.is_file():
                info = json.loads(stats_file.read_text(encoding="utf-8"))
                self.log("Dùng lại mẫu khảo sát đã tải (cache).")
            else:
                scans = self._load_scans()
                infos = self.source.list_files()
                from .hf_parquet import FileMeta

                plan = plan_sample([FileMeta.from_json(s["meta"]) for s in scans], self.cfg)
                strategy = dict(plan.strategy)
                self.log("Chiến lược lấy mẫu: " + json.dumps(strategy, ensure_ascii=False))
                self.log(f"  sẽ tải khoảng {strategy['estimated_download_bytes'] / 1e9:.2f} GB "
                         f"(trần {self.cfg.max_download_gb} GB).")
                frame, stats = fetch_sample(
                    self.source, infos, plan, cache_dir=self.sample_cache, cfg=self.cfg,
                    exclude_ids=self.bronze.ids if self.bronze else ())
                self.sample_cache.mkdir(parents=True, exist_ok=True)
                tmp = final.with_name(final.name + ".tmp")
                frame.to_parquet(tmp, index=False)
                tmp.replace(final)
                info = {"strategy": strategy, "stats": stats,
                        "bronze_excluded": self.bronze is not None}
                report.write_json(stats_file, info)
                self._sample = frame
            sample = self.sample
            counts = sample["domain"].fillna(NULL_LABEL).value_counts().to_dict()
            table = profiling.counts_table(counts, self.known_labels)
            self._save_csv(table, "survey_sample_label_counts")
            full = self._full_counts()
            labels = self._compare_labels(full, counts)
            comparison, stats = profiling.compare_distributions(full, counts, labels)
            self._save_csv(comparison, "survey_sample_vs_full")
            profiling.fig_paired_distribution(
                comparison, self.figure_path("fig_survey_sample_vs_full.png"),
                reference_name="Toàn bộ dataset", other_name="Mẫu khảo sát",
                title="Mẫu khảo sát ngẫu nhiên so với toàn bộ dataset (% bài)")
            counts_stats = info["stats"]
            return {"survey_sample": {
                "rows_final": int(len(sample)),
                "strategy": info["strategy"],
                "bronze_excluded": info["bronze_excluded"],
                "excluded_bronze_ids": counts_stats["excluded_bronze_ids"],
                "duplicate_ids_removed": counts_stats["duplicate_ids_removed"],
                "skipped_invalid_rows": counts_stats["skipped_invalid_rows"],
                "rows_read_from_row_groups": counts_stats["rows_read"],
                "tvd_vs_full": stats["total_variation_distance"],
                "max_abs_gap_pct_points": stats["max_abs_gap_pct_points"],
                "chi2": stats["chi2"], "p_value": stats["p_value"],
                "http_this_session": counts_stats.get("http"),
            }}

        self._step("sample", "tùy tốc độ mạng: thường 3–15 phút lần đầu (tải ~2–4 GB), "
                   "lần sau đọc cache vài giây", work)

    # ------------------------------------------------------------ 5.4 profile

    def step_profile(self) -> None:
        def work() -> dict[str, Any]:
            sample = self.sample
            labels = [x for x in self.known_labels if (sample["domain"] == x).any()]
            extra = sorted(set(sample["domain"].dropna()) - set(labels) - {""})
            labels += extra
            lengths = profiling.length_profile(sample, labels)
            self._save_csv(lengths, "length_profile")
            profiling.fig_length_by_label(sample, labels, self.figure_path("fig_length_by_label.png"))
            table, dup = noise.conflicting_duplicates(sample)
            self._save_csv(table, "conflicting_duplicates")
            return {
                "conflicting_duplicates": dup,
                "length_overall": {
                    "share_gt_256_tieng": float((sample["n_tokens"] > 256).mean()),
                    "share_gt_512_tieng": float((sample["n_tokens"] > 512).mean()),
                    "median_tieng": float(sample["n_tokens"].median()),
                },
            }

        self._step("profile", "vài chục giây", work)

    # ------------------------------------------------- 5.5 out-of-fold model

    def _analysis_subset(self) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
        return confusion.select_analysis_subset(self.sample, self.canonical, self.cfg)

    def step_model(self) -> None:
        def work() -> dict[str, Any]:
            cache = self.model_cache
            subset, classes, skipped = self._analysis_subset()
            y = subset["domain"].map({c: i for i, c in enumerate(classes)}).to_numpy()
            counts = subset["domain"].value_counts().reindex(classes)
            self._save_csv(pd.DataFrame({"label": classes, "n_docs": counts.to_numpy()}),
                           "analysis_set_counts")
            self.log(f"Tập phân tích: {len(subset):,} bài, {len(classes)} nhãn "
                     f"(mỗi nhãn tối đa {self.cfg.per_class_cap:,}).")
            self.log("  ước tính: TF-IDF 1–3 phút, mỗi lượt out-of-fold vài phút tới ~20 phút trên Colab CPU.")
            t0 = time.perf_counter()
            vectorizer, X = confusion.vectorize(subset["text"].tolist(), self.cfg, self.cfg.text_char_limit)
            self.log(f"  TF-IDF xong: {X.shape[0]:,} bài × {X.shape[1]:,} đặc trưng "
                     f"({time.perf_counter() - t0:,.0f}s)")
            ids = subset["id"].tolist()
            del subset
            self._release_sample()  # the 300k-article frame is not needed while fitting
            if (cache / "oof_proba.npy").is_file():
                proba = np.load(cache / "oof_proba.npy")
                self.log("  dùng lại xác suất out-of-fold đã cache.")
            else:
                proba = confusion.oof_probabilities(X, y, len(classes), self.cfg, label="đầy đủ")
                cache.mkdir(parents=True, exist_ok=True)
                np.save(cache / "oof_proba.npy", proba)
            oof = confusion.oof_frame(ids, y, proba, classes)
            cosine = confusion.centroid_cosine(X, y, len(classes))
            terms = profiling.top_terms_by_label(X, y, vectorizer.get_feature_names_out(), classes)
            self._save_csv(terms, "top_terms_by_label")
            cache.mkdir(parents=True, exist_ok=True)
            oof.to_parquet(cache / "oof.parquet", index=False)
            np.save(cache / "centroid_cosine.npy", cosine)
            report.write_json(cache / "classes.json", {"classes": classes, "skipped": skipped})
            return {"model_setup": {"n_docs": len(ids), "n_classes": len(classes),
                                    "skipped_small_labels": skipped, "tfidf_features": int(X.shape[1])}}

        self._step("model", "10–30 phút lần đầu trên Colab CPU (5 lần fit Logistic Regression); "
                   "lần sau vài giây", work)

    def _load_model(self) -> dict[str, Any]:
        if self._model is None:
            meta = json.loads((self.model_cache / "classes.json").read_text(encoding="utf-8"))
            self._model = {
                "classes": meta["classes"],
                "oof": pd.read_parquet(self.model_cache / "oof.parquet"),
                "cosine": np.load(self.model_cache / "centroid_cosine.npy"),
            }
        return self._model

    def step_head_vs_full(self) -> None:
        def work() -> dict[str, Any]:
            if not self.cfg.run_head_vs_full:
                return {"head_vs_full": {}}
            model = self._load_model()
            classes = model["classes"]
            subset, same, _ = self._analysis_subset()
            if same != classes:
                raise AssertionError("Tập phân tích thay đổi giữa các bước")
            y = subset["domain"].map({c: i for i, c in enumerate(classes)}).to_numpy()
            cache = self.model_cache / "oof_head.npy"
            if cache.is_file():
                proba = np.load(cache)
            else:
                _v, X = confusion.vectorize(subset["text"].tolist(), self.cfg, self.cfg.head_chars)
                del subset
                self._release_sample()
                proba = confusion.oof_probabilities(X, y, len(classes), self.cfg, label="đầu bài")
                np.save(cache, proba)
            pred = proba.argmax(axis=1)
            head_table, head_overall = confusion.per_label_metrics(y, pred, classes)
            full = model["oof"]
            full_table, full_overall = confusion.per_label_metrics(
                full["given_idx"].to_numpy(), full["top1_idx"].to_numpy(), classes)
            result = pd.DataFrame({
                "label": classes, "f1_full": full_table["f1"], "f1_head": head_table["f1"],
            })
            result["delta_head_minus_full"] = result["f1_head"] - result["f1_full"]
            self._save_csv(result.sort_values("delta_head_minus_full"), "head_vs_full")
            return {"head_vs_full": {
                "macro_f1_head": head_overall["macro_f1"], "macro_f1_full": full_overall["macro_f1"],
                "accuracy_head": head_overall["accuracy"], "accuracy_full": full_overall["accuracy"],
                "head_chars": self.cfg.head_chars, "full_chars": self.cfg.text_char_limit}}

        self._step("head_vs_full", "bằng khoảng một nửa thời gian bước 'model' (bài ngắn hơn); "
                   "tắt bằng RUN_HEAD_VS_FULL=False", work)

    def step_confusion(self) -> None:
        def work() -> dict[str, Any]:
            model = self._load_model()
            classes, oof = model["classes"], model["oof"]
            k = len(classes)
            y, pred = oof["given_idx"].to_numpy(), oof["top1_idx"].to_numpy()
            counts = confusion.confusion_counts(y, pred, k)
            normalized = confusion.row_normalize(counts)
            sym = confusion.symmetric_confusion(normalized)
            tree, order = confusion.label_linkage(sym)
            names = [classes[i] for i in order]
            ordered = pd.DataFrame(normalized[np.ix_(order, order)], index=names, columns=names)
            ordered.index.name = "given_label"
            ordered.to_csv(self.out_dir / "confusion_matrix_oof.csv", encoding="utf-8")
            confusion.fig_confusion_matrix(normalized, classes, order, self.figure_path("fig_confusion_matrix_oof.png"))
            pairs = confusion.pair_table(counts, oof, model["cosine"], classes)
            self._save_csv(pairs, "confusing_pairs")
            confusion.fig_top_confusing_pairs(pairs, self.cfg.top_pairs, self.figure_path("fig_top_confusing_pairs.png"))
            confusion.fig_dendrogram(tree, classes, self.figure_path("fig_label_dendrogram.png"))
            metrics, overall = confusion.per_label_metrics(y, pred, classes)
            self._save_csv(metrics, "per_label_metrics_oof")
            confusion.fig_per_label_f1(metrics, self.figure_path("fig_per_label_f1.png"))
            return {
                "model": {**overall, "n_docs": int(len(oof)), "per_class_cap": self.cfg.per_class_cap,
                          "n_classes": k, "n_folds": self.cfg.n_folds},
                "dendrogram_order": names,
            }

        self._step("confusion", "vài giây", work)

    # ------------------------------------------------------- 5.6 / 5.7 / 5.8

    def step_noise(self) -> None:
        def work() -> dict[str, Any]:
            model = self._load_model()
            table = noise.mismatch_table(model["oof"], model["classes"], self.cfg.mismatch_prob)
            self._save_csv(table, "suspected_mislabels_by_label")
            examples = noise.suspected_examples(model["oof"], self.cfg.mismatch_prob)
            self._save_csv(examples, "suspected_mislabels_ids")
            return {"suspected_mislabels": {
                "threshold": self.cfg.mismatch_prob, "n_flagged": int(len(examples)),
                "share_flagged": float(len(examples) / len(model["oof"]))}}

        self._step("noise", "vài giây", work)

    def step_scenarios(self) -> None:
        def work() -> dict[str, Any]:
            model = self._load_model()
            full = self.table("full_label_counts")
            full_pct = {str(r.label): float(r.pct) for r in full.itertuples()}
            pairs = self.table("confusing_pairs")
            table, curve = scenarios.run_scenarios(
                model["oof"], model["classes"], pairs, full_pct=full_pct, top_pairs=self.cfg.top_pairs)
            self._save_csv(table, "label_scenarios")
            self._save_csv(curve, "greedy_merge_curve")
            confusion.fig_greedy_merge_curve(curve, self.figure_path("fig_greedy_merge_curve.png"))
            return {"scenarios": {"n": int(len(table))}}

        self._step("scenarios", "vài giây", work)

    def step_audit(self) -> None:
        def work() -> dict[str, Any]:
            model = self._load_model()
            pairs = self.table("confusing_pairs")
            texts = self.sample.set_index("id")["text"]
            pack = audit.build_audit_pack(model["oof"], texts, pairs, self.cfg)
            audit.write_audit_pack(pack, self.out_dir / "audit_pack.csv")
            (self.out_dir / "audit_instructions.md").write_text(audit.AUDIT_INSTRUCTIONS, encoding="utf-8")
            return {"audit_pack": {"rows": int(len(pack)),
                                   "by_part": {k: int(v) for k, v in pack["part"].value_counts().items()}}}

        self._step("audit", "vài giây", work)

    # ---------------------------------------------------------------- finalize

    def run_all(self) -> Path:
        if not self.complete:
            self.step_full_counts()
            self.step_bronze_compare()
            self.step_sample()
            self.step_profile()
            self.step_model()
            self.step_head_vs_full()
            self.step_confusion()
            self.step_noise()
            self.step_scenarios()
            self.step_audit()
        return self.finalize()

    def finalize(self) -> Path:
        """Write summary + report + completion marker (once), then (re)build the zip."""
        if self.complete:
            self.log(f"Thư mục {self.out_dir.name} đã hoàn tất từ trước: không ghi đè, chỉ đóng zip lại.")
            return report.make_zip(self.out_dir, self.zip_path)
        self.log("▶ [report] viết summary.json và báo cáo")
        for marker in sorted((self.out_dir / "steps").glob("*.json")):
            data = json.loads(marker.read_text(encoding="utf-8"))
            self.summary.update(data["summary"])
            self.timings[marker.stem] = data["seconds"]
        from dataclasses import asdict

        summary: dict[str, Any] = {
            "run_id": self.run_id,
            "config": asdict(self.cfg),
            "revision": self.cfg.hf_revision,
            "bronze_manifest": self.bronze.manifest_name if self.bronze else None,
            **self.summary,
            "timings_seconds": self.timings,
            "limitations": LIMITATIONS,
            "environment": {
                "python": platform.python_version(),
                "libraries": {p: _version(p) for p in (
                    "numpy", "pandas", "scikit-learn", "scipy", "pyarrow", "matplotlib",
                    "huggingface_hub", "joblib")},
                "repo_git_commit": _git_commit(self.repo_dir),
            },
            "note": "Không chứa văn bản bài viết.",
        }
        tables = {}
        for name in ("full_label_counts", "label_position_drift", "bronze_vs_full", "snapshot_vs_full",
                     "length_profile", "top_terms_by_label", "confusing_pairs", "per_label_metrics_oof",
                     "suspected_mislabels_by_label", "label_scenarios"):
            path = self.out_dir / f"{name}.csv"
            if path.is_file():
                tables[name] = pd.read_csv(path)
        report.write_json(self.out_dir / "summary.json", summary)
        files = sorted(
            str(p.relative_to(self.out_dir)) for p in self.out_dir.rglob("*")
            if p.is_file() and "steps" not in p.parts and not p.name.endswith(".tmp")
        ) + ["label_survey_report.md"]
        text = report.render_report(summary, tables, sorted(set(files)))
        (self.out_dir / "label_survey_report.md").write_text(text, encoding="utf-8")
        report.mark_complete(self.out_dir, self.run_id)
        self.complete = True
        self.log(f"✔ Đã viết báo cáo: {self.out_dir / 'label_survey_report.md'}")
        return report.make_zip(self.out_dir, self.zip_path)
