"""Offline tests for the Bronze News EDA / TF-IDF / clustering modules (fake Data Lake)."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("scipy")
pytest.importorskip("matplotlib")
pytest.importorskip("yaml")

import bronze_news_fakes as fakes  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from vietnamese_topic_classification.analysis.bronze_news import (  # noqa: E402
    clustering,
    eda,
    features,
    interpret,
    loading,
    output,
    plots,
    report,
    shape,
)
from vietnamese_topic_classification.analysis.bronze_news.config import (  # noqa: E402
    BronzeNewsConfig,
)
from vietnamese_topic_classification.analysis.news_v3 import load_v3_labels  # noqa: E402

CFG = BronzeNewsConfig(
    min_df=2, svd_components=30, k_range=(3, 8), main_min_clusters=3, tsne_max_points=500,
    near_dup_sample=500, neighbor_per_label=40, slow_max_points=200, stability_seeds=3,
    knn_k=10, keywords_per_cluster=15,
)
EXPECTED_LABEL = {
    "politics": {"News_Politics"}, "accident": {"News_Crime_Disaster"}, "sports": {"Sports"},
    "economy": {"Business_and_Industrial", "Finance"},
}


@pytest.fixture(scope="module")
def lake(tmp_path_factory):
    root = tmp_path_factory.mktemp("lake")
    info = fakes.write_lake(root)
    return root, info


def snapshot(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def pipeline(lake):
    root, info = lake
    data = loading.load_bronze_news(root, CFG, log=lambda *_: None)
    frame = eda.prepare_frame(data.news)
    corpus, dropped = eda.drop_summary(frame, True)
    space = features.fit_feature_space(corpus["norm"].tolist(), CFG)
    corpus = corpus[space.kept].reset_index(drop=True)
    X = space.X_lsa
    metrics, arrays, curve = shape.diagnose(X, CFG)
    profile = shape.shape_profile(metrics)
    table, labels, plan = clustering.run_comparison(X, CFG, metrics, profile, log=lambda *_: None)
    ranked = clustering.rank_configs(table, CFG)
    main = clustering.choose_main(ranked, X, CFG)
    truth = [i.split("-")[1] if i.startswith("news-") and i.split("-")[1] in fakes.TOPICS else "x"
             for i in corpus["id"]]
    return dict(data=data, frame=frame, corpus=corpus, dropped=dropped, space=space, X=X,
                metrics=metrics, profile=profile, table=table, labels=labels, plan=plan,
                ranked=ranked, main=main, truth=np.array(truth), arrays=arrays, curve=curve)


# ------------------------------------------------------------------ config / run_id

def test_run_id_is_stable_and_depends_on_seed_config_and_bronze():
    sha = "a" * 64
    assert CFG.run_id(sha) == BronzeNewsConfig.from_mapping(
        {"MIN_DF": 2, "SVD_COMPONENTS": 30, "K_RANGE": [3, 8], "MAIN_MIN_CLUSTERS": 3,
         "TSNE_MAX_POINTS": 500, "NEAR_DUP_SAMPLE": 500, "NEIGHBOR_PER_LABEL": 40,
         "SLOW_MAX_POINTS": 200, "STABILITY_SEEDS": 3, "KNN_K": 10,
         "KEYWORDS_PER_CLUSTER": 15, "UNKNOWN_KEY": 1}).run_id(sha)
    assert CFG.run_id(sha).startswith("bronze_news_aaaaaaaa_seed42_")
    others = {BronzeNewsConfig(seed=1).run_id(sha), BronzeNewsConfig().run_id(sha),
              BronzeNewsConfig().run_id("b" * 64)}
    assert len(others) == 3
    # time budget / n_jobs do not change the result, so they must not change the id
    assert BronzeNewsConfig(time_budget_seconds=5, n_jobs=3).run_id(sha) == \
        BronzeNewsConfig().run_id(sha)


def test_config_validation_and_algorithm_dict():
    with pytest.raises(ValueError):
        BronzeNewsConfig(k_range=(1, 5))
    with pytest.raises(ValueError):
        BronzeNewsConfig(algorithms=("nope",))
    cfg = BronzeNewsConfig.from_mapping({"ALGORITHMS": {"kmeans": True, "gmm": False}})
    assert cfg.algorithms == ("kmeans",)


# ------------------------------------------------------------------ loading / EDA

def test_loading_counts_news_and_samples_neighbors(lake, pipeline):
    root, info = lake
    before = snapshot(root)
    data = loading.load_bronze_news(root, CFG, log=lambda *_: None)
    assert snapshot(root) == before  # read-only
    assert data.manifest_name == "source_manifest_seed42_n50000.json"
    assert data.total_records == len(info["rows"]) and data.sha256 == info["sha256"]
    assert len(data.news) == 4 * 90 + 7  # topics + the 7 injected defects
    assert data.label_counts["News"] == len(data.news) and data.label_counts["<null>"] == 1
    sizes = data.neighbors["domain"].value_counts()
    assert set(sizes.index) == set(CFG.neighbor_labels) and (sizes == 40).all()
    again = loading.load_bronze_news(root, CFG, log=lambda *_: None)
    assert list(again.neighbors["id"]) == list(data.neighbors["id"])  # seeded reservoir


def test_manifest_can_be_chosen_explicitly(lake):
    root, _ = lake
    with pytest.raises(FileNotFoundError):
        loading.load_bronze_news(root, CFG, "missing.json", log=lambda *_: None)


def test_quality_counts_are_exact(pipeline):
    q = eda.quality_report(pipeline["frame"])
    assert q["news_articles"] == 367
    assert q["text_null"] == 1 and q["text_empty_or_blank"] == 1 and q["text_unusable_total"] == 2
    assert q["id_duplicated_rows"] == 2 and q["id_duplicated_distinct"] == 1
    assert q["exact_duplicate_groups"] == 1 and q["exact_duplicate_extra_copies"] == 3
    assert pipeline["dropped"] == {"input": 367, "dropped_empty": 2,
                                   "dropped_exact_duplicates": 3, "kept": 362}
    kept, info = eda.drop_summary(pipeline["frame"], False)
    assert info["dropped_exact_duplicates"] == 0 and len(kept) == 365


def test_length_noise_and_vocabulary(pipeline):
    frame = pipeline["frame"]
    usable = frame[~frame["is_empty"]]
    lengths = eda.length_table(usable["norm"])
    stats = eda.length_stats(lengths, short_words=30, long_words=3000)
    assert stats["words"]["p50"] == pytest.approx(40, abs=2) and stats["long_documents"] == 0
    flags = eda.noise_flags(usable["norm"])
    assert flags["has_url"].sum() == 1 and flags["has_email"].sum() == 1
    assert flags["has_phone"].sum() == 1
    terms = eda.top_terms(usable["text"].tolist(), top=10)
    assert len(terms["unigram"]) == 10 and terms["bigram"]["rate"].max() <= 1.0
    assert not set(terms["unigram"]["term"]) & features.STOPWORDS
    noise = eda.noise_report(flags, [("câu lặp", 9)])
    assert noise["has_url"] == 1 and noise["boilerplate_max_documents"] == 9


def test_near_duplicates_are_reported_not_removed():
    texts = ["bóng đá cầu thủ trận đấu giải đấu bàn thắng"] * 2 + [
        f"chính phủ quốc hội đại biểu số{chr(97 + i)} bầu cử" for i in range(10)]
    rep = eda.near_duplicate_report(texts, threshold=0.99, sample_size=100, seed=1)
    assert rep["pairs"] >= 1 and rep["documents_involved"] >= 2


def test_compare_with_neighbors_reuses_keyword_report(pipeline):
    news = pipeline["frame"].query("~is_empty")["text"].tolist()
    result = eda.compare_news_with_neighbors(news, pipeline["data"].neighbors, top=8)
    sports = result["labels"]["Sports"]
    assert set(sports["shared_terms"]) & set(fakes.FILLER)  # words used by every label
    assert set(sports["neighbor_specific_terms"]) & set(fakes.TOPICS["sports"])
    law = result["labels"]["Law_and_Government"]
    assert set(law["neighbor_specific_terms"]) & set(fakes.TOPICS["law"])
    assert set(law["news_specific_terms"]) & set(fakes.TOPICS["accident"])
    assert 0.0 <= sports["vocabulary_jaccard"] <= 1.0
    assert "text_excerpt" not in json.dumps(result)


# ------------------------------------------------------------------ features / shape

def test_tfidf_lsa_shapes(pipeline):
    space, corpus = pipeline["space"], pipeline["corpus"]
    assert space.X_tfidf.shape[0] == len(corpus) == pipeline["X"].shape[0] == 362
    assert pipeline["X"].shape[1] == 30
    assert np.allclose(np.linalg.norm(pipeline["X"], axis=1), 1.0, atol=1e-6)
    stats = space.matrix_stats()
    assert 0 < stats["tfidf_density"] < 0.2 and stats["lsa_explained_variance"] <= 1.0
    _, lsa, has = space.transform(pipeline["data"].neighbors["text"].tolist())
    assert lsa.shape == (len(pipeline["data"].neighbors), 30) and has.all()


def test_tokenizer_cleans_urls_numbers_and_stopwords():
    tokens = features.tokenize("Xem https://x.vn/a  abc@d.com ngày 12/3 BẦU CỬ của những người")
    assert "bầu" in tokens and "cử" in tokens
    assert not {"https", "của", "những"} & set(tokens)
    assert not any(ch.isdigit() for t in tokens for ch in t)


def test_hopkins_separates_structure_from_noise():
    rng = np.random.default_rng(0)
    blobs = np.vstack([rng.normal(c, 0.05, size=(100, 10)) for c in (0, 1, 2, 3)])
    noise = rng.normal(size=(400, 10))
    assert shape.hopkins(blobs, seed=1, null="uniform")[0] > 0.8
    assert 0.35 < shape.hopkins(noise, seed=1, null="permute")[0] < 0.65


def test_shape_diagnostics(pipeline):
    m = pipeline["metrics"]
    assert m["kmeans_best_k_silhouette"] == 4 and m["gmm_best_k_bic"] in (3, 4, 5)
    assert 0 <= m["hopkins"] <= 1 and m["kdist_knee_eps"] > 0
    assert (np.diff(pipeline["arrays"]["kdistance"]) >= 0).all()
    assert pipeline["profile"]["separation"] == "rõ"
    text = "\n".join(shape.interpret_shape(m, pipeline["profile"]))
    assert "Hopkins" in text and "silhouette" in text
    coords, rows = shape.project_tsne(pipeline["X"], seed=42, max_points=200)
    again, _ = shape.project_tsne(pipeline["X"], seed=42, max_points=200)
    assert coords.shape == (200, 2) and np.allclose(coords, again) and len(rows) == 200


# ------------------------------------------------------------------ clustering

def test_plan_skips_with_reasons(pipeline):
    plan = {p["algorithm"]: p for p in pipeline["plan"]}
    assert plan["meanshift"]["run"] is False and "chiều" in plan["meanshift"]["reason"]
    off = clustering.plan_algorithms(
        BronzeNewsConfig(algorithms=("kmeans",)), pipeline["profile"], 100, 10)
    assert [p["algorithm"] for p in off if p["run"]] == ["kmeans"]
    assert all(p["reason"] for p in off)
    table = pipeline["table"]
    assert set(table["status"]) <= {"ok", "bỏ qua", "lỗi", "không hội tụ"}
    assert "lỗi" not in set(table["status"])


def test_clusters_recover_planted_topics(pipeline):
    assert len(pipeline["main"]) == 2
    assert len({m["algorithm"] for m in pipeline["main"]}) == 2
    truth = pipeline["truth"]
    for pick in pipeline["main"]:
        labels = pipeline["labels"][pick["config_id"]]
        mask = truth != "x"
        assert adjusted_rand_score(truth[mask], labels[mask]) > 0.9, pick["config_id"]
        assert pick["stability_ari"] >= CFG.main_min_stability and pick["criteria"]
    cols = ["n_clusters", "noise_frac", "silhouette_cosine", "davies_bouldin",
            "calinski_harabasz", "seconds"]
    assert set(cols) <= set(pipeline["table"].columns)
    ari, nmi = clustering.agreement(pipeline["labels"], [m["config_id"] for m in pipeline["main"]])
    assert ari.iloc[0, 1] > 0.9 and nmi.iloc[0, 1] > 0.9


def test_density_algorithms_report_noise_fraction(pipeline):
    t = pipeline["table"]
    dens = t[t["algorithm"].isin(["hdbscan", "dbscan", "optics"]) & (t["status"] == "ok")]
    assert len(dens) and dens["noise_frac"].between(0, 1).all()


def test_slow_algorithms_use_a_bounded_sample(pipeline):
    t = pipeline["table"]
    optics = t[(t["algorithm"] == "optics") & (t["status"] == "ok")]
    assert (optics["n_fit"] == 200).all()  # SLOW_MAX_POINTS cap
    for cid in optics["config_id"]:
        assert len(pipeline["labels"][cid]) == 362  # extended to every article


def test_time_budget_skips_slow_configs(pipeline):
    cfg = BronzeNewsConfig(**{**CFG.__dict__, "time_budget_seconds": 1e-9,
                              "algorithms": ("optics", "spectral")})
    table, labels, _ = clustering.run_comparison(
        pipeline["X"], cfg, pipeline["metrics"], pipeline["profile"], log=lambda *_: None)
    assert (table["status"] == "bỏ qua").sum() >= 1


def test_kmeans_stability_curve_and_seed_determinism(pipeline):
    curve = clustering.kmeans_stability_curve(pipeline["X"], CFG)
    assert curve["ari_mean"].between(0, 1).all() and len(curve) == 6
    a, _ = clustering.run_config("kmeans", {"k": 4}, pipeline["X"], CFG)
    b, _ = clustering.run_config("kmeans", {"k": 4}, pipeline["X"], CFG)
    assert (a == b).all()


# ------------------------------------------------------------------ interpretation

def test_seed_keywords_use_v3_labels():
    seeds = interpret.load_seed_keywords()
    assert set(seeds) <= set(load_v3_labels())
    assert {"News_Politics", "News_Crime_Disaster", "News_Current_Affairs", "Sports",
            "Law_and_Government", "Business_and_Industrial", "Finance", "Health"} <= set(seeds)
    assert all(terms for terms in seeds.values())


def test_keywords_and_label_hints_point_the_right_way(pipeline):
    main = pipeline["main"][0]
    labels = pipeline["labels"][main["config_id"]]
    kw = interpret.cluster_keywords(pipeline["space"].X_tfidf, labels,
                                    pipeline["space"].feature_names, 15)
    hints = interpret.suggest_labels(kw, interpret.load_seed_keywords())
    truth = pipeline["truth"]
    seen = set()
    for c, terms in kw.items():
        topic = Counter(truth[labels == c]).most_common(1)[0][0]
        if topic == "x":
            continue
        seen.add(topic)
        vocab = set(fakes.TOPICS[topic])
        words = {w for t, _ in terms for w in t.split()}
        assert len(words & vocab) >= 8, (topic, terms[:8])
        assert hints[c] and hints[c][0]["label"] in EXPECTED_LABEL[topic], (topic, hints[c])
        assert hints[c][0]["score"] > 0
    assert seen == set(EXPECTED_LABEL)
    assert report.LIMITATIONS and interpret.HINT_NOTE.startswith("gợi ý")


def test_representatives_are_closest_to_the_centroid(pipeline):
    labels = pipeline["labels"][pipeline["main"][0]["config_id"]]
    dist = interpret.distance_to_centroid(pipeline["X"], labels)
    reps = interpret.representatives(dist, labels, 3)
    for c in interpret.cluster_ids(labels):
        d = dist[labels == c]
        centre = reps[(reps["cluster"] == c) & (reps["kind"] == "centre")]
        assert centre["distance"].max() <= np.sort(d)[2] + 1e-12
        edge = reps[(reps["cluster"] == c) & (reps["kind"] == "edge")]
        assert edge["distance"].min() >= np.sort(d)[-3] - 1e-12


def test_neighbor_overlap_finds_the_grey_zone(pipeline):
    labels = pipeline["labels"][pipeline["main"][0]["config_id"]]
    data, space = pipeline["data"], pipeline["space"]
    _, neigh_lsa, has = space.transform(data.neighbors["text"].tolist())
    overlap, majority = interpret.neighbor_overlap(
        pipeline["X"], labels, neigh_lsa[has], data.neighbors["domain"][has].tolist(), k=10)
    truth = pipeline["truth"]
    by_topic = {}
    for row in overlap.itertuples():
        topic = Counter(truth[labels == row.cluster]).most_common(1)[0][0]
        by_topic[topic] = row
    assert by_topic["sports"].nearest_label_by_centroid == "Sports"
    assert by_topic["economy"].nearest_label_by_centroid == "Business_and_Industrial"
    assert by_topic["sports"].leak_rate > 0.5 > by_topic["politics"].leak_rate
    assert by_topic["sports"].leak_top_label == "Sports"
    shares = overlap[[c for c in overlap.columns if c.startswith("knn_share_")]].sum(axis=1)
    assert np.allclose(shares, 1.0) and len(majority) == len(labels)


def test_length_profile_flags_short_clusters():
    labels = np.array([0] * 10 + [1] * 10)
    lengths = pd.DataFrame({"chars": [50] * 10 + [900] * 10, "words": [8] * 10 + [150] * 10})
    noise = pd.DataFrame({"has_url": [False] * 20, "has_email": [False] * 20})
    prof = interpret.length_by_cluster(lengths, noise, labels, short_words=30)
    assert prof["artifact_flag"].tolist() == [True, False]


# ------------------------------------------------------------------ output

def test_output_root_never_uses_or_writes_the_lake(tmp_path):
    lake_root, drive = tmp_path / "lake", tmp_path / "MyDrive"
    lake_root.mkdir()
    drive.mkdir()
    root, reason = output.choose_output_root(lake_root, drive, tmp_path / "local")
    assert root == drive / output.MY_DRIVE_FOLDER and "My Drive" in reason
    assert list(lake_root.iterdir()) == []  # not even a write probe
    with pytest.raises(ValueError):
        output.choose_output_root(lake_root, drive, tmp_path / "local",
                                  override=lake_root / "04_reports")
    root, _ = output.choose_output_root(lake_root, None, tmp_path / "local")
    assert root == tmp_path / "local"
    root, why = output.choose_output_root(lake_root, drive, tmp_path / "x",
                                          override=tmp_path / "chosen")
    assert root == tmp_path / "chosen" and why == "OUTPUT_ROOT_OVERRIDE"


def test_export_has_no_article_text_and_never_overwrites(lake, pipeline, tmp_path):
    root, info = lake
    before = snapshot(root)
    data, corpus, labels_all = pipeline["data"], pipeline["corpus"], pipeline["labels"]
    pick = pipeline["main"][0]
    labels = labels_all[pick["config_id"]]
    dist = interpret.distance_to_centroid(pipeline["X"], labels)
    kw = interpret.cluster_keywords(pipeline["space"].X_tfidf, labels,
                                    pipeline["space"].feature_names, 10)
    hints = interpret.suggest_labels(kw, interpret.load_seed_keywords())
    lengths = eda.length_table(corpus["norm"])
    noise = eda.noise_flags(corpus["norm"])
    summary_df = interpret.build_cluster_summary(
        labels, kw, hints, None, interpret.length_by_cluster(lengths, noise, labels, 30))
    reps = interpret.representatives(dist, labels, 2)
    internal = pd.DataFrame({
        "cluster": reps["cluster"], "id": corpus["id"].to_numpy()[reps["row"]],
        "excerpt": [interpret.excerpt(corpus["text"].iloc[r], 500) for r in reps["row"]]})
    q = eda.quality_report(pipeline["frame"])
    summary = {
        "run_id": "r", "config": {"seed": 42}, "bronze": {"manifest": "m", "sha256": "0" * 64,
                                                          "records": data.total_records},
        "eda": {"quality": q, "near_duplicate": {"sample_size": 1, "threshold": 0.9, "pairs": 0},
                "length": eda.length_stats(lengths, short_words=30, long_words=3000),
                "noise": eda.noise_report(noise, [])},
        "clustering_input": pipeline["dropped"], "features": pipeline["space"].matrix_stats(),
        "shape": {"interpretation": shape.interpret_shape(pipeline["metrics"], pipeline["profile"]),
                  "thresholds": shape.THRESHOLDS, "metrics": pipeline["metrics"]},
        "algorithm_plan": pipeline["plan"], "main_configs": pipeline["main"], "agreement": None,
    }
    md = report.render_report(summary, summary_df)
    assert "Cụm 0" in md and "không phải nhãn" in md and "Giới hạn" in md
    store = plots.FigureStore(show=False)
    store.add("t.png", plots.cluster_sizes(summary_df, "t"))
    guard = output.TextGuard(list(data.news["text"]) + list(data.neighbors["text"]))
    out_dir = tmp_path / "out" / "bronze_news_eda" / "r"
    assignments = pd.DataFrame({"id": corpus["id"], "cluster": labels, "dist": dist})
    written = output.export_results(
        out_dir, "r", tables={"cluster_assignments.csv": assignments,
                              "cluster_summary.csv": interpret.flatten_summary(summary_df)},
        jsons={"cluster_summary.json": summary_df.to_dict(orient="records"),
               "eda_summary.json": summary}, figures=store, internal=internal, report_md=md,
        guard=guard)
    assert {"_COMPLETE.json", "report.md", "figures/t.png", output.INTERNAL_NAME} <= set(written)
    # no article text anywhere except the INTERNAL file
    probes = [t[:60] for t in data.news["text"].dropna() if t.strip()][:200]
    for path in out_dir.rglob("*"):
        if not path.is_file() or path.suffix == ".png" or path.name == output.INTERNAL_NAME:
            continue
        content = path.read_text(encoding="utf-8")
        assert not any(p in content for p in probes), path.name
    assert any(p[:30] in (out_dir / output.INTERNAL_NAME).read_text(encoding="utf-8")
               for p in probes)
    json.loads((out_dir / "eda_summary.json").read_text(encoding="utf-8"))  # strict JSON
    # a complete directory is never rewritten
    marker = (out_dir / "_COMPLETE.json").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        output.export_results(out_dir, "r", tables={}, jsons={}, figures=store, internal=internal,
                              report_md="", guard=guard)
    assert (out_dir / "_COMPLETE.json").read_text(encoding="utf-8") == marker
    assert snapshot(root) == before  # the lake was not touched


def test_text_guard_rejects_article_text(pipeline):
    texts = pipeline["data"].news["text"].dropna().tolist()
    guard = output.TextGuard(texts)
    guard.check({"keywords": ["quốc hội", "bầu cử"]}, "ok")
    with pytest.raises(ValueError):
        guard.check({"x": [texts[0]]}, "bad")
    with pytest.raises(ValueError):
        guard.check_frame(pd.DataFrame({"c": [texts[1]]}), "bad")
    with pytest.raises(ValueError):
        guard.check("x" * 2000, "long")
