"""Offline tests for the survey maths: confusion, scenarios, noise, audit pack, config."""
from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from scipy.cluster import hierarchy  # noqa: E402

from vietnamese_topic_classification.analysis.label_survey import (  # noqa: E402
    audit,
    confusion,
    noise,
    profiling,
    scenarios,
)
from vietnamese_topic_classification.analysis.label_survey.config import (  # noqa: E402
    SurveyConfig,
)


def oof_from_confusion(classes, counts, top2=None):
    """Build an out-of-fold frame whose confusion matrix equals ``counts`` exactly."""
    rows = []
    for i, row in enumerate(counts):
        for j, n in enumerate(row):
            for _ in range(n):
                second = (j + 1) % len(classes) if top2 is None else top2(i, j)
                rows.append((i, j, second))
    n = len(rows)
    given = np.array([r[0] for r in rows])
    t1 = np.array([r[1] for r in rows])
    t2 = np.array([r[2] for r in rows])
    names = np.array(classes, dtype=object)
    return pd.DataFrame({
        "id": [f"d{i}" for i in range(n)], "given_idx": given, "given": names[given],
        "given_prob": 0.5, "top1_idx": t1, "top1": names[t1], "top1_prob": 0.8,
        "top2_idx": t2, "top2": names[t2], "top2_prob": 0.1,
    })


def test_symmetric_confusion_and_pair_ranking() -> None:
    classes = ["A", "B", "C"]
    counts = np.array([[8, 2, 0], [1, 9, 0], [0, 0, 10]])
    normalized = confusion.row_normalize(counts)
    sym = confusion.symmetric_confusion(normalized)
    assert sym[0, 1] == pytest.approx((0.2 + 0.1) / 2)
    assert sym[0, 1] == sym[1, 0] and sym[0, 0] == 0
    oof = oof_from_confusion(classes, counts)
    pairs = confusion.pair_table(counts, oof, np.eye(3), classes)
    assert list(pairs.loc[0, ["label_a", "label_b"]]) == ["A", "B"]
    assert pairs.loc[0, "rank"] == 1 and len(pairs) == 3
    assert pairs.loc[0, "rate_a_to_b"] == pytest.approx(0.2)
    assert pairs.loc[0, "n_b_to_a"] == 1
    assert {"sym_confusion", "top2_ambiguity", "centroid_cosine"} <= set(pairs.columns)


def test_top2_ambiguity_needs_close_probabilities_and_given_label_in_the_pair() -> None:
    frame = pd.DataFrame({
        "given_idx": [0, 0, 1, 2], "top1_idx": [0, 1, 0, 0], "top2_idx": [1, 0, 1, 1],
        "top1_prob": [0.45, 0.60, 0.50, 0.45], "top2_prob": [0.40, 0.10, 0.45, 0.40],
    })
    counts, support = confusion.top2_ambiguity(frame, 3, margin=0.2)
    # docs 0 and 2 are close (<0.2) with given in {0,1}; doc 1 is not close;
    # doc 3 is close but its given label (2) is not part of the pair {0,1}
    assert counts[0, 1] == 2 and counts.sum() == 2
    assert list(support) == [2, 1, 1]


def test_centroid_cosine_is_one_for_identical_label_content() -> None:
    from scipy import sparse

    X = sparse.csr_matrix(np.array([[1.0, 0], [2.0, 0], [1.0, 0], [0, 1.0]]))
    y = np.array([0, 0, 1, 2])
    cosine = confusion.centroid_cosine(X, y, 3)
    assert cosine[0, 1] == pytest.approx(1.0) and cosine[0, 2] == pytest.approx(0.0)


def test_dendrogram_groups_the_confusable_labels_first() -> None:
    sym = np.zeros((4, 4))
    sym[0, 1] = sym[1, 0] = 0.3
    sym[2, 3] = sym[3, 2] = 0.2
    tree, order = confusion.label_linkage(sym)
    assert sorted(order) == [0, 1, 2, 3]
    clusters = hierarchy.fcluster(tree, t=0.5, criterion="distance")
    assert clusters[0] == clusters[1] and clusters[2] == clusters[3] and clusters[0] != clusters[2]
    # nothing confused at all must not crash
    confusion.label_linkage(np.zeros((3, 3)))


def test_per_label_metrics_matches_hand_computation() -> None:
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 1])
    table, overall = confusion.per_label_metrics(y, pred, ["A", "B"])
    assert overall["accuracy"] == 0.75
    assert table.loc[0, "precision"] == 1.0 and table.loc[0, "recall"] == 0.5
    assert overall["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)


def merge_fixture():
    """A<->B is the top pair; after merging them C is closer to {A,B} than to D."""
    classes = ["A", "B", "C", "D"]
    counts = np.array([
        [66, 30, 4, 0],     # A -> B 30%
        [30, 66, 4, 0],     # B -> A 30%
        [12, 12, 64, 12],   # C -> A 12%, C -> B 12%, C -> D 12%
        [0, 0, 12, 88],     # D -> C 12%
    ])
    return classes, counts, oof_from_confusion(classes, counts)


def test_greedy_merge_recomputes_confusion_after_every_merge() -> None:
    classes, counts, oof = merge_fixture()
    pairs = confusion.pair_table(counts, oof, np.eye(4), classes)
    # on the ORIGINAL matrix the runner-up pair is C-D (sym .12) ahead of C-A (.08)
    assert (pairs.loc[1, "label_a"], pairs.loc[1, "label_b"]) == ("C", "D")
    table, curve = scenarios.run_scenarios(
        oof, classes, pairs, full_pct={}, top_pairs=2, greedy_steps=2)
    greedy = table[table["scenario"].str.startswith("S2")]
    assert greedy.iloc[0]["merge_order"] == "A + B"
    # after A+B exists, C->(A+B) is 24% vs C->D 12%: the second merge must pull in C
    assert curve["merged"].tolist()[1:] == ["A+B", "A+B+C"]
    assert curve["n_labels"].tolist() == [4, 3, 2]
    assert curve["macro_f1"].is_monotonic_increasing


def test_drop_scenario_uses_top2_for_documents_predicted_as_the_dropped_label() -> None:
    classes = ["A", "B", "C"]
    # (given, top1, top2): the B doc predicted as C has B as its second guess
    rows = [(0, 0, 1), (0, 0, 1), (1, 1, 0), (1, 2, 1), (2, 2, 0), (2, 2, 0)]
    oof = pd.DataFrame({
        "given_idx": [r[0] for r in rows], "top1_idx": [r[1] for r in rows],
        "top2_idx": [r[2] for r in rows],
    })
    pairs = pd.DataFrame({"label_a": ["A"], "label_b": ["B"]})
    table, _ = scenarios.run_scenarios(
        oof, classes, pairs, full_pct={"C": 12.5, "A": 40.0, "B": 47.5}, top_pairs=1,
        greedy_steps=0, lowest_f1_to_drop=3, extra_drop=())
    drop = table[table["scenario"] == "S3: bỏ C"].iloc[0]
    assert drop["accuracy"] == 1.0 and drop["macro_f1"] == 1.0 and drop["n_labels"] == 2
    assert drop["pct_full_dataset_affected"] == 12.5
    assert drop["pct_remaining_docs_repredicted"] == pytest.approx(25.0)
    base = table[table["scenario"] == "S0"].iloc[0]
    assert base["accuracy"] == pytest.approx(5 / 6) and base["affected_pct_analysis_set"] == 0


def test_mismatch_table_and_suspected_examples() -> None:
    classes = ["A", "B"]
    oof = pd.DataFrame({
        "id": list("wxyz"), "given_idx": [0, 0, 0, 1], "given": ["A", "A", "A", "B"],
        "given_prob": [0.1, 0.5, 0.9, 0.9], "top1_idx": [1, 1, 0, 1], "top1": ["B", "B", "A", "B"],
        "top1_prob": [0.85, 0.65, 0.9, 0.9], "top2_idx": [0, 0, 1, 0], "top2": ["A", "A", "B", "A"],
        "top2_prob": [0.1, 0.3, 0.05, 0.05],
    })
    table = noise.mismatch_table(oof, classes, 0.7)
    row = table.set_index("label").loc["A"]
    assert row["n"] == 3 and row["n_confident_mismatch"] == 1  # 0.65 is below the threshold
    assert row["most_often_read_as"] == "B"
    assert list(noise.suspected_examples(oof, 0.7)["id"]) == ["w"]
    assert "text" not in noise.suspected_examples(oof, 0.7).columns


def test_conflicting_duplicates_carry_only_ids_and_labels() -> None:
    sample = pd.DataFrame({
        "id": ["1", "2", "3", "4", "5", "6"],
        "domain": ["A", "B", "A", "A", None, "B"],
        "text_sha": ["h1", "h1", "h2", "h2", "h3", "h3"],
        "text": ["secret"] * 6,
    })
    table, summary = noise.conflicting_duplicates(sample)
    assert list(table.columns) == ["group", "id", "domain"]
    assert sorted(table["id"]) == ["1", "2"]          # h2 agrees, h3 has a null label
    assert summary["conflicting_groups"] == 1 and summary["exact_duplicate_groups"] >= 2


def audit_fixture(n_per_label=40):
    classes = ["A", "B", "C", "D"]
    rng = np.random.default_rng(0)
    counts = np.array([[28, 10, 1, 1]] * 2 + [[1, 1, 30, 8]] * 1 + [[1, 1, 8, 30]])
    counts = counts + 0 * n_per_label
    oof = oof_from_confusion(classes, counts.tolist())
    oof["top1_prob"] = rng.uniform(0.5, 0.99, len(oof))
    texts = {i: ("=SUM(1,1) " + "dòng một\n\ndòng hai " * 100) if k % 3 == 0 else f"nội dung {i}"
             for k, i in enumerate(oof["id"])}
    pairs = pd.DataFrame({"label_a": ["A", "C"], "label_b": ["B", "D"]})
    return oof, texts, pairs


def test_audit_pack_columns_limits_bom_and_excerpt(tmp_path) -> None:
    oof, texts, pairs = audit_fixture()
    cfg = SurveyConfig(top_pairs=2, audit_per_class=5, audit_per_pair=6, audit_part_c=7,
                       mismatch_prob=0.7)
    pack = audit.build_audit_pack(oof, texts, pairs, cfg)
    assert list(pack.columns) == audit.AUDIT_COLUMNS
    assert pack["id"].is_unique and pack["audit_id"].is_unique
    assert (pack["part"] == "C").sum() <= 7
    assert (pack["part"] == "B").sum() <= 2 * 6
    assert pack[pack["part"] == "A"].groupby("given_label").size().max() <= 5
    assert pack["excerpt"].str.len().max() <= 601  # 600 characters (+ optional guard quote)
    assert not pack["excerpt"].str.contains(r"\n").any()
    assert (pack[["reviewer", "reviewer_label", "agrees_with_given", "note"]] == "").all().all()
    path = audit.write_audit_pack(pack, tmp_path / "audit_pack.csv")
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    back = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    assert "nội dung" in " ".join(back["excerpt"]) and len(back) == len(pack)
    assert not back["excerpt"].str.startswith("=").any()
    # part B only holds documents misread inside the listed pairs
    part_b = pack[pack["part"] == "B"]
    pairs_set = {frozenset(p) for p in [("A", "B"), ("C", "D")]}
    assert all(frozenset((g, m)) in pairs_set for g, m in zip(part_b["given_label"], part_b["model_top1"]))


def test_audit_excerpt_neutralises_spreadsheet_formulas() -> None:
    assert audit.make_excerpt("=HYPERLINK('x')").startswith("'=")
    assert audit.make_excerpt("  a \n\n b  ") == "a b"


def test_compare_distributions_tvd_and_chi_square() -> None:
    table, stats = profiling.compare_distributions({"A": 50, "B": 50}, {"A": 60, "B": 40}, ["A", "B"])
    assert stats["total_variation_distance"] == pytest.approx(0.1)
    assert stats["max_abs_gap_pct_points"] == pytest.approx(10.0)
    assert table.set_index("label").loc["A", "diff_pct_points"] == pytest.approx(10.0)
    _t, same = profiling.compare_distributions({"A": 5, "B": 5}, {"A": 50, "B": 50}, ["A", "B"])
    assert same["total_variation_distance"] == 0 and same["p_value"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        profiling.compare_distributions({"A": 0}, {"A": 1}, ["A"])


def test_length_profile_and_top_terms() -> None:
    from scipy import sparse

    sample = pd.DataFrame({"domain": ["A"] * 4 + ["B"] * 2, "n_chars": [10, 20, 30, 40, 5, 5],
                           "n_tokens": [100, 300, 600, 900, 1, 2]})
    table = profiling.length_profile(sample, ["A", "B"]).set_index("label")
    assert table.loc["A", "share_gt_512_tieng"] == 0.5 and table.loc["A", "share_gt_256_tieng"] == 0.75
    assert table.loc["B", "tieng_p50"] == pytest.approx(1.5)
    X = sparse.csr_matrix(np.array([[1.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 1.0, 1.0]]))
    terms = profiling.top_terms_by_label(X, np.array([0, 0, 1, 1]),
                                         np.array(["alpha", "beta", "gamma"]), ["A", "B"], top_k=2)
    assert terms[terms["label"] == "A"].iloc[0]["term"] == "alpha"
    assert terms[terms["label"] == "B"].iloc[0]["term"] in {"beta", "gamma"}


def test_counts_table_flags_unexpected_values() -> None:
    table = profiling.counts_table({"A": 5, "Adult": 1, "<null>": 2, "<empty>": 1}, ["A", "B"])
    kinds = dict(zip(table["label"], table["kind"]))
    assert kinds == {"A": "known", "B": "known", "Adult": "unknown",
                     "<null>": "null_value", "<empty>": "empty_value"}
    assert profiling.max_min_ratio(profiling.counts_table({"A": 10, "B": 2}, ["A", "B"])) == 5.0


def test_config_validation_hash_and_run_id() -> None:
    cfg = SurveyConfig()
    assert cfg.run_id().startswith("survey_b81fcce5_seed42_n300000_")
    assert len(cfg.run_id().rsplit("_", 1)[1]) == 8
    assert cfg.config_hash() == SurveyConfig(http_workers=3, n_jobs=1).config_hash()  # speed only
    assert cfg.config_hash() != SurveyConfig(seed=1).config_hash()
    assert cfg.config_hash({"bronze_sha256": "x"}) != cfg.config_hash({"bronze_sha256": "y"})
    assert cfg.sample_hash() == SurveyConfig(per_class_cap=10).sample_hash()  # modelling-only
    for bad in ({"hf_revision": "main"}, {"seed": -1}, {"n_folds": 1}, {"mismatch_prob": 1.2},
                {"head_chars": 9_000}, {"per_class_cap": 0}, {"max_download_gb": 0}):
        with pytest.raises(ValueError):
            SurveyConfig(**bad)
    built = SurveyConfig.from_mapping({"SEED": 7, "N_FOLDS": 3, "UNRELATED": 1, "np": None})
    assert built.seed == 7 and built.n_folds == 3
