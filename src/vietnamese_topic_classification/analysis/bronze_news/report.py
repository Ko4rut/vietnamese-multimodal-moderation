"""Facts-only Vietnamese summary (markdown). Numbers and keywords only: no article text."""

from __future__ import annotations

from typing import Any

import pandas as pd

LIMITATIONS = (
    "TF-IDF chỉ thấy mặt chữ (từ/cụm từ), không hiểu nghĩa: hai bài cùng chủ đề nhưng khác từ ngữ "
    "có thể rơi vào hai cụm, và hai bài khác chủ đề nhưng dùng nhiều từ chung có thể vào một cụm.",
    "Phân cụm không phải là gán nhãn. Cụm và 'gợi ý nhãn V3' chỉ là điểm xuất phát để người đọc "
    "kiểm tra; quyết định taxonomy thuộc về nhóm.",
    "Nhãn `domain` do máy gán (nvidia/domain-classifier), nên 'nhãn láng giềng' cũng có nhiễu.",
    "Không gian TF-IDF/SVD được fit chỉ trên bài News; bài của nhãn láng giềng được chiếu vào đó "
    "(từ chỉ có ở nhãn láng giềng bị bỏ qua), nên độ giống là ước lượng.",
    "Số cụm, thuật toán chính và ngưỡng chẩn đoán là lựa chọn có tiêu chí nhưng vẫn là heuristic; "
    "kết quả phụ thuộc cấu hình và seed.",
    "Dữ liệu là mẫu Bronze (seed 42), không phải toàn bộ dataset.",
)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render_report(summary: dict[str, Any], clusters: pd.DataFrame) -> str:
    """Build report.md from the summary dict and the cluster summary table."""
    s, q = summary, summary["eda"]["quality"]
    L = summary["eda"]["length"]
    lines = [
        "# Báo cáo EDA + phân cụm cho nhãn `News` trong Bronze (V3)",
        "",
        f"- Run: `{s['run_id']}` | seed {s['config']['seed']}",
        f"- Bronze: `{s['bronze']['manifest']}`, sha256 `{s['bronze']['sha256'][:12]}…`, "
        f"{s['bronze']['records']:,} bản ghi.",
        f"- Số bài `News`: **{q['news_articles']:,}** "
        f"({_pct(q['news_articles'] / max(1, s['bronze']['records']))} của Bronze).",
        "",
        "## Chất lượng dữ liệu",
        f"- Text null/rỗng: {q['text_unusable_total']:,}; `id` trùng: {q['id_duplicated_rows']:,} dòng; "
        f"text trùng chính xác: {q['exact_duplicate_extra_copies']:,} bản sao thừa "
        f"({q['exact_duplicate_groups']:,} nhóm).",
        f"- Near-duplicate (mẫu {summary['eda']['near_duplicate']['sample_size']:,} bài, cosine ≥ "
        f"{summary['eda']['near_duplicate']['threshold']}): "
        f"{summary['eda']['near_duplicate']['pairs']:,} cặp.",
        f"- Dùng cho phân cụm: {s['clustering_input']['kept']:,} bài (bỏ "
        f"{s['clustering_input']['dropped_empty']:,} bài rỗng và "
        f"{s['clustering_input']['dropped_exact_duplicates']:,} bản sao trùng chính xác).",
        f"- Độ dài (số từ): trung vị {L['words']['p50']:,.0f}, p90 {L['words']['p90']:,.0f}, "
        f"p99 {L['words']['p99']:,.0f}; ngắn (<{L['short_threshold_words']} từ) "
        f"{L['short_documents']:,}; dài (>{L['long_threshold_words']} từ) {L['long_documents']:,}.",
        f"- Dấu hiệu nhiễu crawl: URL {summary['eda']['noise']['has_url']:,}, email "
        f"{summary['eda']['noise']['has_email']:,}, số điện thoại {summary['eda']['noise']['has_phone']:,}, "
        f"ký tự lạ {summary['eda']['noise']['odd_chars']:,}; câu lặp giữa nhiều bài: "
        f"{summary['eda']['noise']['boilerplate_sentences_repeated']:,}.",
        "",
        "## Đặc trưng",
        f"- TF-IDF: {s['features']['tfidf_features']:,} đặc trưng, độ dày "
        f"{100 * s['features']['tfidf_density']:.2f}%; LSA {s['features']['lsa_dimensions']} chiều giữ "
        f"{_pct(s['features']['lsa_explained_variance'])} phương sai.",
        "",
        "## Shape của dữ liệu",
    ]
    lines += s["shape"]["interpretation"]
    lines += ["", f"*{s['shape']['thresholds']}*", "", "## Thuật toán", ""]
    plan = pd.DataFrame(s["algorithm_plan"])
    for row in plan.itertuples():
        lines.append(f"- `{row.algorithm}`: {'chạy' if row.run else 'bỏ qua'} — {row.reason}")
    lines += ["", "### Cấu hình chính"]
    for pick in s["main_configs"]:
        lines.append(
            f"- `{pick['config_id']}`: {pick['n_clusters']} cụm, silhouette "
            f"{pick['silhouette_cosine']:.3f}, ổn định (ARI) {pick['stability_ari']:.2f}. "
            f"Tiêu chí: {pick['criteria']}.")
    agreement = s.get("agreement")
    if agreement:
        lines.append(f"- Độ đồng thuận giữa các cấu hình chính: ARI = {agreement['ari']:.2f}, "
                     f"NMI = {agreement['nmi']:.2f}.")
    lines += ["", "## Cụm (cấu hình chính đầu tiên)", ""]
    for r in clusters.itertuples():
        hints = "; ".join(f"{h['label']} ({h['score']:.2f})" for h in r.label_hints) or "không có"
        lines.append(f"### Cụm {r.cluster}: {r.size:,} bài ({r.percent:.1f}%)")
        lines.append(f"- Từ khóa: {', '.join(r.keywords[:12])}")
        lines.append(f"- Gợi ý nhãn V3 (cần người kiểm tra, không phải nhãn): {hints}")
        if getattr(r, "nearest_neighbor_label", ""):
            lines.append(
                f"- Giống nhất nhãn láng giềng: {r.nearest_neighbor_label} "
                f"(cosine tâm {r.nearest_centroid_cosine:.2f}); tỉ lệ bài lọt vào vùng nhãn khác "
                f"{_pct(r.leak_rate)}" + (f" (chủ yếu {r.leak_top_label})" if r.leak_top_label else ""))
        if r.artifact_flag:
            lines.append("- ⚠️ Cụm có dấu hiệu do độ dài/nhiễu crawl hơn là chủ đề.")
        lines.append("")
    lines += ["## Giới hạn"] + [f"- {text}" for text in LIMITATIONS]
    lines += ["", "*Báo cáo chỉ nêu số liệu và từ khóa, không chứa văn bản bài viết và không "
              "quyết định thay nhóm.*", ""]
    return "\n".join(lines)
