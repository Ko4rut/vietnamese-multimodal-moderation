"""Generate the Colab notebook for the Bronze ``News`` EDA + TF-IDF + clustering (stdlib only).

Edit the cells here, then run ``python scripts/16_generate_bronze_news_eda_notebook.py``.
``--check`` fails when ``notebooks/v3/eda/01_bronze_news_eda_colab.ipynb`` is out of date.
Cells tagged ``colab-only`` need Colab (git clone, pip, Drive mount); the offline test runs
every other cell against a fake Data Lake.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1] / "notebooks/v3/eda/01_bronze_news_eda_colab.ipynb"
)

INTRO = """# EDA + TF-IDF + phân cụm cho nhãn `News` trong Bronze (V3)

**Chỉ cần: Runtime → Run all**, rồi bấm cho phép khi Colab hỏi quyền Google Drive (chọn tài khoản có Data Lake). Không cần nhập gì.

## Notebook làm gì?
Nhãn `News` (do máy gán) chứa lẫn nhiều chủ đề con: chính trị, tai nạn/tội phạm, thời sự, thể thao, pháp luật... nên mô hình khó phân biệt `News` với nhãn khác. Notebook này **chỉ khảo sát**, trả lời bằng số liệu:

1. Có bao nhiêu bài `News` trong Bronze, dữ liệu sạch cỡ nào?
2. Dùng **TF-IDF + giảm chiều (LSA)**, dữ liệu có "hình dạng" gì? Thuật toán phân cụm nào hợp?
3. So sánh nhiều thuật toán phân cụm. Mỗi cụm là chủ đề gì (từ khóa, bài mẫu, gợi ý nhãn V3)?
4. **Cụm News nào trông giống `Law_and_Government`, `Sports`...?** (vùng xám)

**Notebook không gán nhãn và không quyết định thay nhóm.** Phân cụm chỉ là gợi ý để người đọc kiểm tra.

## An toàn
- Data Lake chỉ được **đọc** (quyền Viewer là đủ). Kết quả ghi vào **My Drive của bạn**, thư mục `vmm_outputs/bronze_news_eda/<run_id>/` (hoặc `/content` nếu không có Drive).
- Thư mục đã có `_COMPLETE.json` thì **không bao giờ bị ghi đè**.
- Chỉ file `cluster_excerpts_INTERNAL.csv` chứa đoạn trích bài báo. **Chỉ dùng nội bộ, không đưa lên Git** (license dataset chưa xác minh).
- Chạy trên Colab miễn phí (CPU), mất khoảng **10–15 phút**. Seed cố định nên chạy lại ra cùng kết quả."""

CONFIG = '''# ===== Tất cả tham số nằm ở đây. Mặc định đã ổn: không cần sửa gì rồi chạy tiếp. =====
GIT_REF = "feat/bronze-news-eda-v3"   # nhánh code để clone
SEED = 42                             # seed cho mọi bước ngẫu nhiên

USE_DRIVE = True                  # False: không mount Drive, phải đặt DATA_LAKE_ROOT_OVERRIDE
DATA_LAKE_ROOT_OVERRIDE = None    # None: tự tìm "My Drive/vietnamese-multilabel-topic-classification"
BRONZE_MANIFEST_NAME = None       # None: tự tìm manifest Bronze 50k seed 42 và in ra bản đã chọn
OUTPUT_ROOT_OVERRIDE = None       # None: My Drive/vmm_outputs (rồi /content); hoặc đường dẫn bạn ghi được
MOUNT = "/content/vtsnlp_drive"

NEWS_LABEL = "News"
INCLUDE_NEIGHBOR_LABELS = True    # so News với các nhãn láng giềng (mục 6)
NEIGHBOR_LABELS = ("Law_and_Government", "Sensitive_Subjects", "People_and_Society",
                   "Business_and_Industrial", "Sports")
NEIGHBOR_PER_LABEL = 600          # mẫu tối đa mỗi nhãn láng giềng (theo seed)

DROP_EXACT_DUPLICATES = True      # bỏ bản sao trùng chính xác khi phân cụm (EDA vẫn báo cáo đủ)
NEAR_DUP_SAMPLE = 3000            # số bài tối đa khi dò near-duplicate
NEAR_DUP_THRESHOLD = 0.9          # cosine TF-IDF >= ngưỡng này coi là gần trùng
SHORT_WORDS = 30                  # bài ngắn hơn số từ này bị đánh dấu "quá ngắn"
LONG_WORDS = 3000                 # bài dài hơn số từ này bị đánh dấu "quá dài"

# TF-IDF giữ gần baseline (ngram (1,2), sublinear), min_df = 3 thay vì 2 vì ~3000 bài: bỏ bớt cụm quá hiếm
TEXT_CHAR_LIMIT = 8000
NGRAM_RANGE = (1, 2)
MIN_DF = 3
MAX_DF = 0.6
MAX_FEATURES = 30_000
SVD_COMPONENTS = 100              # số chiều LSA

TSNE_MAX_POINTS = 4000            # t-SNE chỉ để vẽ; tối đa bấy nhiêu điểm
K_RANGE = (3, 15)                 # quét số cụm k
ALGORITHMS = {                    # bật/tắt từng thuật toán
    "kmeans": True, "minibatch_kmeans": True, "gmm": True, "birch": True, "ward": True,
    "agglomerative_average_cosine": True, "agglomerative_complete_cosine": True,
    "hdbscan": True, "dbscan": True, "optics": True,
    "spectral": True, "affinity_propagation": True, "meanshift": True,
}
SLOW_MAX_POINTS = 1500            # thuật toán chậm chỉ chạy trên mẫu tối đa bấy nhiêu bài
TIME_BUDGET_SECONDS = 300         # ngân sách thời gian cho nhóm thuật toán chậm
KEYWORDS_PER_CLUSTER = 20
EXCERPT_CHARS = 500               # độ dài đoạn trích (chỉ hiện trong notebook + file INTERNAL)

EXPORT_RESULTS = True             # ghi file kết quả
AUTO_DOWNLOAD_ZIP = True          # tự tải file zip về máy khi xong

print("Cấu hình đã nạp. Nhánh code:", GIT_REF, "| seed:", SEED)'''

CLONE = '''from pathlib import Path
import subprocess

REPO_URL = "https://github.com/Ko4rut/vietnamese-multimodal-moderation.git"
PROJECT_DIR = Path("/content/vmm_bronze_news_project")

if (PROJECT_DIR / ".git").is_dir():
    dirty = subprocess.run(
        ["git", "-C", str(PROJECT_DIR), "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    )
    if dirty.stdout.strip():
        raise RuntimeError("Repository trên Colab có thay đổi chưa commit; hãy dùng Runtime → Disconnect and delete runtime rồi chạy lại.")
    subprocess.run(["git", "-C", str(PROJECT_DIR), "fetch", "origin", GIT_REF], check=True)
    subprocess.run(["git", "-C", str(PROJECT_DIR), "checkout", "-B", GIT_REF, "FETCH_HEAD"], check=True)
else:
    if PROJECT_DIR.exists() and any(PROJECT_DIR.iterdir()):
        raise RuntimeError(f"{PROJECT_DIR} đã tồn tại nhưng không phải Git repository.")
    result = subprocess.run(
        ["git", "clone", "--single-branch", "--branch", GIT_REF, REPO_URL, str(PROJECT_DIR)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Không clone được nhánh '{GIT_REF}'. Nếu nhánh đã được merge và xóa, đổi GIT_REF thành "
            f"\\"main\\" hoặc \\"v3_test\\" ở cell cấu hình. Chi tiết: {result.stderr.strip()}"
        )

revision = subprocess.check_output(
    ["git", "-C", str(PROJECT_DIR), "rev-parse", "--short", "HEAD"], text=True
).strip()
print(f"Repository: {GIT_REF} @ {revision}")'''

INSTALL = '''import importlib
import subprocess
import sys

# Chỉ cài các thư viện phân tích nhẹ (numpy, pandas, scikit-learn, matplotlib...). Không cần GPU.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-e", ".[survey]"],
    cwd=str(PROJECT_DIR), check=True,
)

SRC_DIR = (PROJECT_DIR / "src").resolve()
if not (SRC_DIR / "vietnamese_topic_classification" / "__init__.py").is_file():
    raise FileNotFoundError(f"Không tìm thấy Python package: {SRC_DIR}")
sys.path[:] = [p for p in sys.path if p != str(SRC_DIR)]
sys.path.insert(0, str(SRC_DIR))

# Colab giữ module cũ giữa các lần checkout: xóa cả cây package để import lại bản mới.
PACKAGE = "vietnamese_topic_classification"
for name in sorted(
    [name for name in sys.modules if name == PACKAGE or name.startswith(PACKAGE + ".")],
    key=len, reverse=True,
):
    del sys.modules[name]
importlib.invalidate_caches()

import sklearn
from vietnamese_topic_classification.analysis.bronze_news.config import BronzeNewsConfig
print("Python:", sys.version.split()[0], "| scikit-learn:", sklearn.__version__)'''

DRIVE = '''MOUNTED = False   # Drive đã mount được chưa
if USE_DRIVE:
    try:
        from google.colab import drive

        drive.mount(MOUNT)   # Colab sẽ hỏi quyền: chọn đúng tài khoản có Data Lake rồi bấm Allow
        MOUNTED = True
    except Exception as exc:
        print("⚠️ Không mount được Google Drive:", f"{type(exc).__name__}: {exc}")
else:
    print("USE_DRIVE=False: không mount Drive.")'''

LAKE = '''from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

from vietnamese_topic_classification.analysis.bronze_news import (
    clustering, eda, features, interpret, loading, output, plots, report, shape,
)
from vietnamese_topic_classification.analysis.bronze_news.config import BronzeNewsConfig
from vietnamese_topic_classification.data.colab_lake import resolve_mounted_lake

pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 200)

cfg = BronzeNewsConfig.from_mapping(globals())   # kiểm tra tham số + tính hash cấu hình

# Data Lake: CHỈ ĐỌC. Nếu không tìm thấy, hàm báo lỗi kèm cách sửa.
DATA_LAKE_ROOT = resolve_mounted_lake(MOUNT, override=DATA_LAKE_ROOT_OVERRIDE)
print("Data Lake (chỉ đọc):", DATA_LAKE_ROOT)

# Kết quả KHÔNG bao giờ ghi vào Data Lake: My Drive của bạn, rồi mới tới /content.
OUTPUT_ROOT, OUTPUT_REASON = output.choose_output_root(
    DATA_LAKE_ROOT,
    Path(MOUNT) / "MyDrive" if MOUNTED else None,
    Path("/content/vmm_outputs"),
    override=OUTPUT_ROOT_OVERRIDE,
)
print("Thư mục kết quả:", OUTPUT_ROOT / output.OUTPUT_SUBDIR)
print("  lý do chọn:", OUTPUT_REASON)
if str(OUTPUT_ROOT).startswith("/content"):
    print("⚠️ /content bị xóa khi Colab ngắt kết nối: hãy tải file zip về ngay khi xong.")

figures = plots.FigureStore(show=True)   # lưu PNG trong bộ nhớ, xuất file ở mục 7
timings = {}'''

S1 = '''import time

t0 = time.perf_counter()
data = loading.load_bronze_news(DATA_LAKE_ROOT, cfg, BRONZE_MANIFEST_NAME)   # verify checksum, chỉ đọc
timings["load_bronze"] = round(time.perf_counter() - t0, 1)

run_id = cfg.run_id(data.sha256)
RUN_DIR = output.run_dir(OUTPUT_ROOT, run_id)
ALREADY_COMPLETE = output.is_complete(RUN_DIR)

n_news = len(data.news)
print(f"Manifest: {data.manifest_name}")
print(f"sha256:   {data.sha256}")
print(f"Tổng số bản ghi Bronze: {data.total_records:,}")
print(f"Số bài nhãn '{cfg.news_label}': {n_news:,} ({100 * n_news / data.total_records:.2f}% Bronze)")
print(f"run_id: {run_id}")
if ALREADY_COMPLETE:
    print("ℹ️ Kết quả của run_id này đã có _COMPLETE.json: sẽ KHÔNG ghi đè (chỉ chạy lại để xem).")

counts = pd.Series(data.label_counts, name="số bài").to_frame()
counts["%"] = (100 * counts["số bài"] / data.total_records).round(2)
counts["là News"] = counts.index == cfg.news_label
display(counts)'''

S2A = '''frame = eda.prepare_frame(data.news)
quality = eda.quality_report(frame)
exact_dup = eda.exact_duplicate_mask(frame)
display(pd.Series(quality, name="giá trị").to_frame())

usable = frame[~frame["is_empty"]].reset_index(drop=True)
unique_texts = frame.loc[~frame["is_empty"] & ~exact_dup, "norm"].tolist()
near_dup = eda.near_duplicate_report(
    unique_texts, threshold=NEAR_DUP_THRESHOLD, sample_size=NEAR_DUP_SAMPLE, seed=SEED)
print("Near-duplicate (chỉ báo cáo, không xóa):", near_dup)

corpus, clustering_input = eda.drop_summary(frame, DROP_EXACT_DUPLICATES)
print("Dùng cho phân cụm:", clustering_input)'''

S2B = '''lengths = eda.length_table(usable["norm"])
length_stats = eda.length_stats(lengths, short_words=SHORT_WORDS, long_words=LONG_WORDS)
display(pd.DataFrame({c: length_stats[c] for c in ("chars", "words")}).T.round(1))
print(f"Bài quá ngắn (<{SHORT_WORDS} từ): {length_stats['short_documents']:,} | "
      f"quá dài (>{LONG_WORDS} từ): {length_stats['long_documents']:,}")
figures.add("eda_length_hist.png", plots.length_histograms(lengths, length_stats))'''

S2C = '''noise_flags = eda.noise_flags(usable["norm"])
boilerplate = eda.boilerplate_sentences(usable["norm"].tolist())
noise = eda.noise_report(noise_flags, boilerplate)
display(pd.Series(noise, name="giá trị").to_frame())
if boilerplate:
    print("Câu lặp lại ở nhiều bài khác nhau (chỉ hiện ở đây, không lưu vào báo cáo):")
    for sentence, n_docs in boilerplate[:5]:
        print(f"  [{n_docs} bài] {sentence[:150]}")'''

S2D = '''terms = eda.top_terms(usable["text"].tolist(), top=cfg.top_terms)
display(pd.concat({k: v.assign(rate=(100 * v["rate"]).round(1)) for k, v in terms.items()}, axis=1))
figures.add("eda_top_terms.png", plots.top_terms_bars(terms, top=20))'''

S2E = '''neighbor_terms = {}
if INCLUDE_NEIGHBOR_LABELS and len(data.neighbors):
    neighbor_terms = eda.compare_news_with_neighbors(usable["text"].tolist(), data.neighbors, top=12)
    rows = [{
        "nhãn láng giềng": label, "số bài mẫu": info["documents"],
        "Jaccard từ vựng": round(info["vocabulary_jaccard"], 3),
        "từ chung": ", ".join(info["shared_terms"][:8]),
        "riêng của News": ", ".join(info["news_specific_terms"][:8]),
        "riêng của láng giềng": ", ".join(info["neighbor_specific_terms"][:8]),
    } for label, info in neighbor_terms["labels"].items()]
    display(pd.DataFrame(rows))
    print("Đây chỉ là gợi ý ban đầu cho vùng xám (tần suất từ), chưa phải kết luận.")
else:
    print("Bỏ qua so sánh láng giềng (INCLUDE_NEIGHBOR_LABELS=False hoặc không có mẫu láng giềng).")'''

S2F = '''# Đoạn trích ngẫu nhiên (theo seed). CHỈ hiện trong notebook, không lưu vào báo cáo.
sample = usable.sample(n=min(5, len(usable)), random_state=SEED)
for row in sample.itertuples():
    print(f"[{row.id}] {interpret.excerpt(row.text, 300)}\\n")'''

S3 = '''t0 = time.perf_counter()
space = features.fit_feature_space(corpus["norm"].tolist(), cfg)
corpus = corpus[space.kept].reset_index(drop=True)   # bỏ bài không còn đặc trưng nào
timings["features"] = round(time.perf_counter() - t0, 1)

feature_stats = space.matrix_stats()
display(pd.Series(feature_stats, name="giá trị").to_frame())
print(f"Ma trận TF-IDF: {space.X_tfidf.shape[0]:,} bài x {space.X_tfidf.shape[1]:,} đặc trưng, "
      f"độ dày {100 * feature_stats['tfidf_density']:.2f}% (còn lại là số 0).")
print(f"Sau LSA: {space.X_lsa.shape[0]:,} x {space.X_lsa.shape[1]}, giữ "
      f"{100 * feature_stats['lsa_explained_variance']:.1f}% phương sai.")
figures.add("tfidf_svd_variance.png", plots.variance_curve(space.explained_variance_ratio))'''

S4A = '''X = space.X_lsa   # không gian phân cụm: LSA đã chuẩn hóa L2 (Euclid ~ cosine)
t0 = time.perf_counter()
shape_metrics, shape_arrays, k_curve = shape.diagnose(X, cfg)
profile = shape.shape_profile(shape_metrics)
pca2 = shape.project_pca(X, SEED)
tsne2, tsne_rows = shape.project_tsne(
    X, seed=SEED, perplexity=cfg.tsne_perplexity, max_points=cfg.tsne_max_points)
timings["shape"] = round(time.perf_counter() - t0, 1)

figures.add("shape_projection.png", plots.projections(pca2[tsne_rows], tsne2))
figures.add("shape_kdistance.png", plots.kdistance_plot(
    shape_arrays["kdistance"], shape_metrics["kdist_k"], shape_metrics["kdist_knee_eps"]))
figures.add("shape_cosine_distance.png", plots.cosine_hist(shape_arrays["cosine_distances"]))
stability_curve = clustering.kmeans_stability_curve(X, cfg)
figures.add("shape_k_curves.png", plots.k_curves_plot(k_curve, stability_curve))
display(k_curve.round(3))'''

S4B = '''shape_text = shape.interpret_shape(shape_metrics, profile)
display(Markdown("### Diễn giải tự động từ số đo\\n\\n" + "\\n".join(shape_text)
                 + "\\n\\n*" + shape.THRESHOLDS + "*"))
print("Hồ sơ shape:", profile)'''

S5A = '''t0 = time.perf_counter()
table, labels_by_cfg, plan = clustering.run_comparison(X, cfg, shape_metrics, profile)
ranked = clustering.rank_configs(table, cfg)
timings["clustering"] = round(time.perf_counter() - t0, 1)

display(pd.DataFrame(plan)[["algorithm", "run", "reason"]])
show = ["config_id", "n_fit", "n_clusters", "noise_frac", "silhouette_cosine", "davies_bouldin",
        "calinski_harabasz", "seconds", "rank_score", "eligible_main"]
display(ranked[ranked["status"] == "ok"][show].head(15).round(3))
bad = table[table["status"] != "ok"][["config_id", "status", "note"]]
if len(bad):
    display(bad)'''

S5B = '''main_configs = clustering.choose_main(ranked, X, cfg)
for pick in main_configs:
    print(f"Cấu hình chính: {pick['config_id']} | {pick['n_clusters']} cụm | silhouette "
          f"{pick['silhouette_cosine']:.3f} | ổn định (ARI) {pick['stability_ari']:.2f}")
    print("   tiêu chí:", pick["criteria"])

# Mỗi thuật toán lấy cấu hình xếp hạng cao nhất để so sánh và vẽ lưới kiểu scikit-learn.
best = ranked[ranked["rank_score"].notna()].drop_duplicates("algorithm")
ids = [c for c in best["config_id"] if c in labels_by_cfg]
agree_ari, agree_nmi = clustering.agreement(labels_by_cfg, ids)
print("\\nARI giữa các thuật toán (cấu hình tốt nhất của mỗi thuật toán):")
display(agree_ari.round(2))
agreement = None
if len(main_configs) >= 2:
    pair = [main_configs[0]["config_id"], main_configs[1]["config_id"]]
    pair_ari, pair_nmi = clustering.agreement(labels_by_cfg, pair)
    agreement = {"ari": float(pair_ari.iloc[0, 1]), "nmi": float(pair_nmi.iloc[0, 1])}
    print(f"Đồng thuận giữa 2 cấu hình chính: ARI={agreement['ari']:.2f}, NMI={agreement['nmi']:.2f}")

panels = []
for row in best.itertuples():
    if row.config_id in labels_by_cfg:
        panels.append((row.algorithm, labels_by_cfg[row.config_id][tsne_rows], float(row.seconds),
                       f"{row.params} | {int(row.n_clusters)} cụm"))
figures.add("cluster_algorithm_grid.png", plots.algorithm_grid(tsne2, panels))
display(stability_curve.round(3))'''

S6A = '''primary = main_configs[0]
labels = labels_by_cfg[primary["config_id"]]
print("Cấu hình phân tích sâu:", primary["config_id"])
dist = interpret.distance_to_centroid(X, labels)
keywords = interpret.cluster_keywords(space.X_tfidf, labels, space.feature_names,
                                      cfg.keywords_per_cluster)
hints = interpret.suggest_labels(keywords, interpret.load_seed_keywords())

corpus_lengths = eda.length_table(corpus["norm"])
corpus_noise = eda.noise_flags(corpus["norm"])
length_profile = interpret.length_by_cluster(corpus_lengths, corpus_noise, labels, SHORT_WORDS)
figures.add("cluster_main_tsne.png", plots.main_clusters_plot(
    tsne2, labels[tsne_rows], primary["config_id"]))'''

S6B = '''overlap = None
neighbor_pack = None
if INCLUDE_NEIGHBOR_LABELS and len(data.neighbors):
    # Chiếu nhãn láng giềng vào CÙNG không gian TF-IDF/SVD đã fit trên News (không fit lại).
    _, neigh_lsa, has_terms = space.transform(data.neighbors["text"].tolist())
    neigh = data.neighbors[has_terms].reset_index(drop=True)
    neigh_lsa = neigh_lsa[has_terms]
    overlap, majority = interpret.neighbor_overlap(
        X, labels, neigh_lsa, neigh["domain"].tolist(), k=cfg.knn_k)
    neighbor_pack = (neigh, neigh_lsa)
    print(f"Mẫu láng giềng: {len(neigh):,} bài ({neigh['domain'].nunique()} nhãn); "
          f"bài News lọt vào vùng nhãn khác (k-NN cân bằng): {100 * (majority != cfg.news_label).mean():.1f}%")
    cols = ["cluster", "size", "nearest_label_by_centroid", "nearest_centroid_cosine", "leak_rate",
            "leak_top_label"] + [c for c in overlap.columns if c.startswith("knn_share_")]
    display(overlap[cols].round(3))
    figures.add("cluster_neighbor_heatmap.png", plots.overlap_heatmap(overlap))
    cn, rn, cg, rg = shape.joint_tsne(X, neigh_lsa, seed=SEED, perplexity=cfg.tsne_perplexity,
                                      max_points=cfg.overlay_max_points)
    figures.add("cluster_neighbor_scatter.png", plots.neighbor_overlay(
        cn, labels[rn], cg, neigh["domain"].to_numpy()[rg]))
else:
    print("Bỏ qua vùng xám (INCLUDE_NEIGHBOR_LABELS=False hoặc không có mẫu láng giềng).")'''

S6C = '''cluster_summary = interpret.build_cluster_summary(labels, keywords, hints, overlap, length_profile)
display(interpret.flatten_summary(cluster_summary)[[
    "cluster", "size", "percent", "keywords", "label_hints", "artifact_flag"]].round(1))
figures.add("cluster_sizes.png", plots.cluster_sizes(cluster_summary, "Số bài mỗi cụm"))
figures.add("cluster_length_by_cluster.png", plots.length_by_cluster_plot(
    corpus_lengths["words"], labels))
display(length_profile.round(3))
n_noise = int((labels < 0).sum())
if n_noise:
    print(f"{n_noise:,} bài ({100 * n_noise / len(labels):.1f}%) là điểm nhiễu của thuật toán mật độ.")'''

S6D = '''reps = interpret.representatives(dist, labels, cfg.representatives_per_cluster)
reps["id"] = corpus["id"].to_numpy()[reps["row"].to_numpy()]
reps["excerpt"] = [interpret.excerpt(corpus["text"].iloc[r], EXCERPT_CHARS) for r in reps["row"]]
internal_excerpts = reps[["cluster", "kind", "rank", "id", "distance", "excerpt"]].assign(
    note="CHỈ DÙNG NỘI BỘ: có đoạn trích bài báo, không đưa lên Git/chia sẻ công khai")

# Chỉ hiện trong notebook (đoạn trích). Mỗi cụm: từ khóa, gợi ý nhãn, bài đại diện, bài ở rìa.
for c in interpret.cluster_ids(labels):
    row = cluster_summary[cluster_summary["cluster"] == c].iloc[0]
    hint = "; ".join(f"{h['label']} ({h['score']:.2f})" for h in row["label_hints"]) or "không có"
    print("=" * 100)
    print(f"CỤM {c}: {row['size']:,} bài ({row['percent']:.1f}%)")
    print("Từ khóa:", ", ".join(row["keywords"][:15]))
    print("Gợi ý nhãn V3 (CẦN NGƯỜI KIỂM TRA, không phải nhãn):", hint)
    if overlap is not None:
        o = overlap[overlap["cluster"] == c].iloc[0]
        print(f"Giống nhất nhãn láng giềng: {o['nearest_label_by_centroid']} "
              f"(cosine tâm {o['nearest_centroid_cosine']:.2f}) | lọt vùng nhãn khác: {100 * o['leak_rate']:.0f}%")
    if row["artifact_flag"]:
        print("⚠️ Cụm này có thể là bài ngắn/nhiễu crawl hơn là một chủ đề.")
    for kind, name in (("centre", "đại diện"), ("edge", "ở rìa")):
        for r in internal_excerpts[(internal_excerpts["cluster"] == c) & (internal_excerpts["kind"] == kind)].head(2).itertuples():
            print(f"  - ({name}) [{r.id}] {r.excerpt[:260]}")'''

S7A = '''import json
from dataclasses import asdict

bad_hints = [c for c, h in hints.items() if not h]
summary = {
    "run_id": run_id,
    "config": asdict(cfg),
    "bronze": {"manifest": data.manifest_name, "sha256": data.sha256, "records": data.total_records,
               "label_counts": data.label_counts},
    "eda": {"quality": quality, "near_duplicate": near_dup, "length": length_stats, "noise": noise,
            "neighbor_vocabulary": neighbor_terms.get("labels", {})},
    "clustering_input": clustering_input,
    "features": feature_stats,
    "shape": {"metrics": shape_metrics, "profile": profile, "interpretation": shape_text,
              "thresholds": shape.THRESHOLDS},
    "algorithm_plan": plan,
    "main_configs": main_configs,
    "agreement": agreement,
    "primary_config": primary["config_id"],
    "clusters_without_label_hint": bad_hints,
    "timings_seconds": timings,
    "limitations": list(report.LIMITATIONS),
    "note": "Không chứa văn bản bài viết. Gợi ý nhãn V3 cần người kiểm tra, không phải nhãn.",
}
report_md = report.render_report(summary, cluster_summary)
display(Markdown(report_md))'''

S7B = '''from vietnamese_topic_classification.analysis.bronze_news.output import TextGuard

assignments = pd.DataFrame({"id": corpus["id"].to_numpy()})
for pick in main_configs:
    assignments["cluster_" + pick["algorithm"]] = labels_by_cfg[pick["config_id"]]
assignments["tsne_x"], assignments["tsne_y"] = np.nan, np.nan
assignments.loc[tsne_rows, "tsne_x"] = tsne2[:, 0]
assignments.loc[tsne_rows, "tsne_y"] = tsne2[:, 1]
assignments["pca_x"], assignments["pca_y"] = pca2[:, 0], pca2[:, 1]
assignments["distance_to_centroid"] = dist   # cosine tới tâm cụm của cấu hình phân tích sâu

zip_path = None
if not EXPORT_RESULTS:
    print("EXPORT_RESULTS=False: không ghi file.")
elif ALREADY_COMPLETE:
    print(f"{RUN_DIR} đã hoàn tất từ trước: KHÔNG ghi đè, chỉ đóng zip lại.")
    zip_path = output.make_zip(RUN_DIR, RUN_DIR.parent / f"{run_id}.zip")
else:
    written = output.export_results(
        RUN_DIR, run_id,
        tables={
            "cluster_assignments.csv": assignments,
            "cluster_summary.csv": interpret.flatten_summary(cluster_summary),
            "algorithm_comparison.csv": ranked,
        },
        jsons={
            "cluster_summary.json": cluster_summary.to_dict(orient="records"),
            "eda_summary.json": summary,
            "run_config.json": {"run_id": run_id, "config": asdict(cfg), "git_ref": GIT_REF},
        },
        figures=figures, internal=internal_excerpts, report_md=report_md,
        guard=TextGuard(list(data.news["text"]) + list(data.neighbors["text"])),
    )
    zip_path = output.make_zip(RUN_DIR, RUN_DIR.parent / f"{run_id}.zip")
    print("Đã ghi", len(written), "file vào:", RUN_DIR)
    for name in written:
        print("  -", name)
if zip_path:
    print("File zip:", zip_path, f"({zip_path.stat().st_size / 1e6:.1f} MB)")
    if AUTO_DOWNLOAD_ZIP:
        output.try_download(zip_path)
print("Thời gian từng bước (giây):", timings)'''


def cells() -> list[tuple[str, str, str, list[str]]]:
    md = lambda cid, text: ("markdown", cid, text, [])  # noqa: E731
    code = lambda cid, src, tags=None: ("code", cid, src, tags or [])  # noqa: E731
    return [
        md("intro", INTRO),
        md("config-md", "## 0. Cấu hình và môi trường\n\nTất cả tham số nằm ở một cell. **Không cần sửa** để chạy bình thường."),
        code("config", CONFIG),
        md("install-md", "Clone repo theo `GIT_REF`, cài thư viện nhẹ (`pip install -e \".[survey]\"`), rồi mount Google Drive. Colab sẽ hỏi quyền: chọn **đúng tài khoản có Data Lake**. Data Lake chỉ được đọc; kết quả ghi vào My Drive của bạn."),
        code("clone", CLONE, ["colab-only"]),
        code("install", INSTALL, ["colab-only"]),
        code("drive", DRIVE, ["colab-only"]),
        code("lake", LAKE),
        md("section-1-md", "## 1. Đọc và kiểm tra Bronze\n\nĐọc Bronze (chỉ đọc), kiểm tra checksum, rồi lọc nhãn `News`. Bảng cuối đặt `News` trong bối cảnh toàn bộ các nhãn. Số bài News được **đếm trực tiếp**, không gán cứng."),
        code("section-1", S1),
        md("section-2-md", "## 2. EDA của bài `News`\n\nChỉ **báo cáo**, không xóa gì, trừ bước chuẩn bị dữ liệu phân cụm: bài rỗng luôn bị bỏ, bản sao trùng chính xác bị bỏ khi `DROP_EXACT_DUPLICATES=True` (in rõ số bài bỏ)."),
        md("s2a-md", "### 2.1 Chất lượng dữ liệu\nText rỗng/null, `id` trùng, text trùng chính xác (sha256 sau chuẩn hóa NFC + khoảng trắng), near-duplicate (cosine TF-IDF trên một mẫu giới hạn)."),
        code("section-2a", S2A),
        md("s2b-md", "### 2.2 Độ dài bài\nĐo bằng số ký tự và số từ (tách theo khoảng trắng, tiếng Việt ≈ số âm tiết). Trục log vì độ dài lệch phải mạnh."),
        code("section-2b", S2B),
        md("s2c-md", "### 2.3 Dấu hiệu nhiễu crawl\nURL, email, số điện thoại, ký tự lạ, và câu lặp lại ở nhiều bài khác nhau (boilerplate)."),
        code("section-2c", S2C),
        md("s2d-md", "### 2.4 Từ vựng\nTừ đơn và cụm 2 âm tiết phổ biến nhất (theo số bài chứa), đã bỏ stopword tiếng Việt."),
        code("section-2d", S2D),
        md("s2e-md", "### 2.5 So sánh nhanh với nhãn láng giềng\nTừ chung và từ riêng giữa `News` và `Law_and_Government`, `Sensitive_Subjects`, `People_and_Society`, `Business_and_Industrial`, `Sports`. Gợi ý ban đầu cho vùng xám."),
        code("section-2e", S2E),
        md("s2f-md", "### 2.6 Vài đoạn trích ngẫu nhiên\nChỉ để nhìn dữ liệu bằng mắt. **Không lưu** vào báo cáo."),
        code("section-2f", S2F),
        md("section-3-md", "## 3. Trích xuất đặc trưng: TF-IDF rồi LSA\n\n**Các bước:** chuẩn hóa NFC, chữ thường, bỏ URL/email/số/ký tự đặc biệt, tách thành âm tiết, bỏ stopword, rồi `TfidfVectorizer` với 1–2 từ liền nhau (giữ cụm như \"bầu cử\", \"tai nạn\"), `sublinear_tf=True`. Cấu hình bám baseline của repo (`ngram_range=(1,2)`); khác ở `min_df=3` (thay vì 2) và `max_df=0.6`, vì với ~3000 bài, từ xuất hiện 2 lần là quá hiếm để làm tâm cụm.\n\n**Vì sao phải giảm chiều?** TF-IDF có hàng chục nghìn cột, gần như toàn số 0. Trong không gian đó, mọi cặp bài đều \"xa nhau như nhau\", nên các thuật toán dựa trên khoảng cách chạy kém. Ta dùng **LSA** (`TruncatedSVD`) nén xuống ~100 chiều rồi chuẩn hóa L2, khi đó khoảng cách Euclid tương đương cosine."),
        code("section-3", S3),
        md("section-4-md", "## 4. Chẩn đoán \"shape\" của dữ liệu\n\nTrước khi chọn thuật toán, xem dữ liệu có xu hướng tạo cụm không, mật độ có đều không, cụm có dạng cầu không (bảng so sánh thuật toán của [scikit-learn](https://scikit-learn.org/stable/modules/clustering.html) chọn theo đúng các tiêu chí này). Các số đo: Hopkins, k-distance, phân bố khoảng cách cosine, elbow/silhouette của KMeans, BIC/AIC của GMM. **Thuật toán không được chọn sẵn**: mục 5 quyết định chạy hay bỏ dựa trên các số đo này."),
        code("section-4a", S4A),
        code("section-4b", S4B),
        md("section-5-md", "## 5. Phân cụm và so sánh thuật toán\n\nChạy trên không gian LSA. Nhóm tâm/phân phối (KMeans, MiniBatchKMeans, GMM, BIRCH), phân cấp (Ward, average/complete + cosine), mật độ (HDBSCAN, DBSCAN, OPTICS), và nhóm khác (Spectral, AffinityPropagation, MeanShift). Thuật toán chậm chạy trên mẫu giới hạn kèm ngân sách thời gian; thuật toán không hợp shape bị **bỏ qua kèm lý do**. Mỗi cấu hình được chấm bằng silhouette (cosine), Davies–Bouldin, Calinski–Harabasz, số cụm, tỉ lệ nhiễu và thời gian. Độ ổn định đo bằng ARI giữa nhiều seed."),
        code("section-5a", S5A),
        code("section-5b", S5B),
        md("section-6-md", "## 6. Diễn giải cụm (phần quan trọng nhất)\n\nVới cấu hình chính: số bài mỗi cụm, từ khóa (TF-IDF trung bình của cụm trừ trung bình toàn bộ, tính trên không gian TF-IDF gốc), bài đại diện/bài ở rìa (chỉ đoạn trích, chỉ trong notebook), **gợi ý nhãn V3** từ bộ keyword hạt giống (`seed_keywords.yaml`, dễ sửa). Gợi ý chỉ là điểm số để người kiểm tra, **không phải nhãn**.\n\n**Vùng xám:** mẫu các nhãn láng giềng được chiếu vào đúng không gian TF-IDF/SVD đã fit trên News (không fit lại, để cụm News nguyên vẹn). Với mỗi cụm tính nhãn láng giềng gần nhất và tỉ lệ bài News mà láng giềng gần nhất chủ yếu thuộc nhãn khác."),
        code("section-6a", S6A),
        code("section-6b", S6B),
        code("section-6c", S6C),
        code("section-6d", S6D),
        md("section-7-md", "## 7. Tóm tắt và xuất file\n\nBản tóm tắt chỉ nêu số liệu, không quyết định thay nhóm. Sau đó ghi `cluster_assignments.csv`, `cluster_summary.csv/json`, `algorithm_comparison.csv`, `eda_summary.json`, `figures/`, `report.md`, `cluster_excerpts_INTERNAL.csv` (**chỉ dùng nội bộ**), `run_config.json` và `_COMPLETE.json` vào thư mục kết quả, rồi đóng zip và tự tải về."),
        code("section-7a", S7A),
        code("section-7b", S7B),
    ]


def build_notebook() -> dict:
    notebook_cells = []
    for kind, cell_id, source, tags in cells():
        cell = {"cell_type": kind, "id": cell_id,
                "metadata": {"tags": tags} if tags else {}, "source": source}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        notebook_cells.append(cell)
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"name": "01_bronze_news_eda_colab.ipynb", "provenance": []},
        },
        "cells": notebook_cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="Fail if the notebook is out of sync")
    args = parser.parse_args()
    serialized = json.dumps(build_notebook(), indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != serialized:
            parser.error(f"Notebook is missing or out of date: {args.output}")
        print("Notebook up to date:", args.output)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print("Generated:", args.output)


if __name__ == "__main__":
    main()
