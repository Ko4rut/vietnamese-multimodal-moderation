"""summary.json, the Vietnamese facts-only report, the zip bundle and the download."""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

COMPLETE_MARKER = "_COMPLETE.json"


def write_json(path: Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=_json_default) + "\n",
                   encoding="utf-8")
    tmp.replace(path)
    return path


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    return str(value)


def is_complete(out_dir: Path) -> bool:
    return (Path(out_dir) / COMPLETE_MARKER).is_file()


def mark_complete(out_dir: Path, run_id: str) -> None:
    write_json(Path(out_dir) / COMPLETE_MARKER,
               {"run_id": run_id, "completed_utc": datetime.now(timezone.utc).isoformat()})


def make_zip(out_dir: Path, zip_path: Path) -> Path:
    """Zip a finished output folder (top-level folder = run id). Never overwrites a zip
    that is different in content silently: an existing zip is replaced atomically."""
    out_dir, zip_path = Path(out_dir), Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = zip_path.with_name(zip_path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file() and not path.name.endswith(".tmp"):
                archive.write(path, Path(out_dir.name) / path.relative_to(out_dir))
    tmp.replace(zip_path)
    return zip_path


def try_download(zip_path: Path) -> bool:
    """Ask the browser to download the zip (Colab). Never raises."""
    try:
        from google.colab import files  # type: ignore[import-not-found]

        files.download(str(zip_path))
        return True
    except Exception as exc:
        print(
            "Không tự tải được file zip (trình duyệt có thể đang chặn tải xuống: "
            f"{type(exc).__name__}).\nFile nằm ở: {zip_path}\n"
            "Tải tay: mở bảng Files bên trái (biểu tượng thư mục) → tìm file → "
            "chuột phải → Download."
        )
        return False


# ------------------------------------------------------------------ markdown report


def md_table(frame: pd.DataFrame, formats: Mapping[str, str] | None = None) -> str:
    if frame is None or frame.empty:
        return "_(không có dữ liệu)_\n"
    formats = formats or {}
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    divider = "|" + "|".join("---" for _ in frame.columns) + "|"
    lines = [header, divider]
    for row in frame.itertuples(index=False):
        cells = []
        for column, value in zip(frame.columns, row):
            if isinstance(value, float) and value != value:
                cells.append("")
            elif column in formats and isinstance(value, (int, float)):
                cells.append(format(value, formats[column]))
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _num(value: Any, spec: str = ",") -> str:
    return "n/a" if value is None else format(value, spec)


def render_report(summary: Mapping[str, Any], tables: Mapping[str, pd.DataFrame],
                  files: Sequence[str]) -> str:
    """Facts and numbers only: no recommendation to keep, merge or drop a label."""
    s = summary
    full, bronze = s.get("full_dataset", {}), s.get("bronze_vs_full", {})
    sample, model = s.get("survey_sample", {}), s.get("model", {})
    pairs = tables.get("confusing_pairs", pd.DataFrame())
    metrics = tables.get("per_label_metrics_oof", pd.DataFrame())
    mismatch = tables.get("suspected_mislabels_by_label", pd.DataFrame())
    dups = s.get("conflicting_duplicates", {})
    head = s.get("head_vs_full", {})

    ten = []
    ten.append(f"Toàn bộ dataset có **{_num(full.get('total_rows'))}** bài trong "
               f"**{_num(full.get('n_files'))}** file, **{_num(full.get('n_row_groups'))}** row group.")
    odd = full.get("unexpected_label_values", [])
    ten.append(f"Có **{full.get('n_label_values')}** giá trị nhãn khác nhau; nhãn không nằm trong "
               f"`domains_v1.yaml` (kể cả null/rỗng/`Adult`): "
               f"**{', '.join(odd) if odd else 'không có'}**.")
    ten.append(f"Nhãn lớn nhất / nhỏ nhất (trong các nhãn chuẩn có mặt) = **{_num(full.get('max_min_ratio'), '.1f')}** lần "
               f"({full.get('largest_label')} / {full.get('smallest_label')}).")
    if bronze.get("status") == "ok":
        ten.append(f"Mẫu Bronze 50k lệch so với toàn bộ: TVD = **{bronze['total_variation_distance']:.4f}**, "
                   f"chênh lớn nhất **{bronze['max_abs_gap_pct_points']:.2f}** điểm %.")
    else:
        ten.append(f"So sánh Bronze 50k: **bỏ qua**, lý do: {bronze.get('reason', 'không có dữ liệu')}.")
    ten.append(f"Mẫu khảo sát ngẫu nhiên: **{_num(sample.get('rows_final'))}** bài; so với toàn bộ "
               f"TVD = **{_num(sample.get('tvd_vs_full'), '.4f')}**.")
    if model:
        ten.append(f"Mô hình phân tích out-of-fold: Macro-F1 = **{model['macro_f1']:.3f}**, "
                   f"accuracy = **{model['accuracy']:.3f}** (trên {_num(model['n_docs'])} bài, "
                   f"mỗi nhãn tối đa {_num(model['per_class_cap'])}).")
    if len(pairs):
        top = pairs.iloc[0]
        ten.append(f"Cặp lẫn nhiều nhất: **{top.label_a} ↔ {top.label_b}** "
                   f"(sym_confusion = {top.sym_confusion:.3f}).")
    if len(metrics):
        low = metrics.sort_values("f1").iloc[0]
        ten.append(f"Nhãn có F1 thấp nhất: **{low.label}** (F1 = {low.f1:.3f}).")
    if len(mismatch):
        top = mismatch.iloc[0]
        ten.append(f"Nhãn có tỉ lệ nghi gán sai cao nhất: **{top.label}** "
                   f"({100 * top.confident_mismatch_rate:.1f}% bài bị đoán thành nhãn khác "
                   f"với xác suất ≥ {s['config']['mismatch_prob']}).")
    ten.append(f"Nhóm bài giống hệt nhau nhưng khác nhãn: **{dups.get('conflicting_groups', 'n/a')}** nhóm "
               f"({dups.get('rows_in_conflicting_groups', 'n/a')} bài) trong mẫu khảo sát.")
    if head:
        ten.append(f"Chỉ đọc {s['config']['head_chars']} ký tự đầu: Macro-F1 = **{head['macro_f1_head']:.3f}** "
                   f"so với **{head['macro_f1_full']:.3f}** khi đọc tới {s['config']['text_char_limit']} ký tự.")

    out = [f"# Báo cáo khảo sát nhãn `domain` — {s['run_id']}\n",
           f"Dataset `{s['config']['hf_dataset']}` @ `{s['config']['hf_revision'][:10]}`. "
           "Báo cáo tự sinh; **chỉ nêu sự thật và con số**, không đưa ra khuyến nghị giữ/gộp/bỏ nhãn. "
           "Nhãn `domain` do máy gán (`nvidia/domain-classifier`), chưa được người kiểm duyệt.\n",
           "## 1. Tóm tắt các con số quan trọng\n"]
    out += [f"{i}. {line}" for i, line in enumerate(ten, 1)]

    out += ["\n## 2. Phân bổ toàn bộ dataset và độ lệch của mẫu 50k\n",
            md_table(tables.get("full_label_counts", pd.DataFrame())[["label", "count", "pct", "rank", "kind"]]
                     if "full_label_counts" in tables else pd.DataFrame(),
                     {"count": ",", "pct": ".2f"})]
    drift = tables.get("label_position_drift", pd.DataFrame())
    if len(drift):
        out += ["**Thay đổi tỉ lệ nhãn theo vị trí file** (5 nhãn dao động nhiều nhất giữa các shard; "
                "độ lệch chuẩn tính theo điểm %):\n", md_table(drift.head(5), {
                    "pct_min": ".2f", "pct_max": ".2f", "pct_std": ".2f", "corr_with_file_index": ".2f"})]
    if bronze.get("status") == "ok":
        out += [f"**Mẫu Bronze 50k so với toàn bộ** (chi-square = {bronze['chi2']:.1f}, "
                f"p = {bronze['p_value']:.3g}; với n lớn p gần như luôn nhỏ nên nhìn độ chênh là chính). "
                "10 nhãn chênh nhiều nhất:\n",
                md_table(tables["bronze_vs_full"].head(10),
                         {"reference_pct": ".2f", "other_pct": ".2f", "diff_pct_points": "+.2f", "ratio": ".2f"})]
    else:
        out += [f"**So sánh Bronze 50k: bỏ qua.** Lý do: {bronze.get('reason', 'không có dữ liệu')}.\n"]
    snaps = tables.get("snapshot_vs_full", pd.DataFrame())
    if len(snaps):
        out += ["**Các snapshot khác** (Silver/Gold train, so với toàn bộ):\n", md_table(snaps, {
            "total_variation_distance": ".4f", "max_abs_gap_pct_points": ".2f"})]
    out += [f"**Mẫu khảo sát** — chiến lược: `{json.dumps(sample.get('strategy', {}), ensure_ascii=False)}`. "
            f"Đã loại {sample.get('excluded_bronze_ids', 'n/a')} bài trùng ID Bronze.\n"]

    out += ["\n## 3. Hồ sơ nhãn\n", md_table(tables.get("length_profile", pd.DataFrame())[
        ["label", "n", "chars_p50", "tieng_p5", "tieng_p50", "tieng_p95", "share_gt_256_tieng", "share_gt_512_tieng"]
    ] if "length_profile" in tables else pd.DataFrame(), {
        "chars_p50": ",.0f", "tieng_p5": ",.0f", "tieng_p50": ",.0f", "tieng_p95": ",.0f",
        "share_gt_256_tieng": ".2f", "share_gt_512_tieng": ".2f"}),
        "Ghi chú: *tiếng* tách theo khoảng trắng, không phải token của mô hình. "
        "Từ đặc trưng từng nhãn nằm trong `top_terms_by_label.csv`.\n"]
    terms = tables.get("top_terms_by_label", pd.DataFrame())
    if len(terms):
        short = (terms[terms["rank"] <= 6].groupby("label", sort=False)["term"]
                 .apply(lambda x: ", ".join(x)).reset_index().rename(columns={"term": "6 từ/cụm đặc trưng nhất"}))
        out += [md_table(short)]

    out += ["\n## 4. Top cặp lẫn và dendrogram\n",
            f"Ba tín hiệu độc lập cho {s['config']['top_pairs']} cặp đầu "
            "(`sym_confusion`, `top2_ambiguity`, `centroid_cosine`):\n",
            md_table(pairs.head(s["config"]["top_pairs"])[
                ["rank", "label_a", "label_b", "sym_confusion", "top2_ambiguity", "centroid_cosine"]
            ] if len(pairs) else pd.DataFrame(), {
                "sym_confusion": ".3f", "top2_ambiguity": ".3f", "centroid_cosine": ".3f"}),
            f"Thứ tự lá trong dendrogram: {' → '.join(s.get('dendrogram_order', []))}\n"]
    if len(metrics):
        out += ["**Hiệu năng từng nhãn (out-of-fold)** — đây là mô hình dùng để phân tích, không phải "
                "mô hình chính thức, và không so trực tiếp với baseline Gold của nhóm:\n",
                md_table(metrics.sort_values("f1"), {"precision": ".3f", "recall": ".3f", "f1": ".3f"})]
    if head:
        out += [f"**Chỉ đọc đầu bài**: Macro-F1 {head['macro_f1_head']:.3f} (đầu {s['config']['head_chars']} "
                f"ký tự) so với {head['macro_f1_full']:.3f} (tới {s['config']['text_char_limit']} ký tự). "
                "Chi tiết từng nhãn: `head_vs_full.csv`.\n"]

    out += ["\n## 5. Nghi gán sai\n",
            f"Bài bị mô hình đoán nhãn khác với xác suất ≥ {s['config']['mismatch_prob']}. "
            "Đây là **nghi vấn**, không phải kết luận; mô hình cũng có thể sai.\n",
            md_table(mismatch, {"confident_mismatch_rate": ".3f", "most_often_read_as_share": ".2f",
                                "mean_prob_of_given_label": ".3f"})]

    scen = tables.get("label_scenarios", pd.DataFrame())
    out += ["\n## 6. Kịch bản \"nếu thì\" (chỉ để tham khảo)\n",
            md_table(scen[["scenario", "n_labels", "macro_f1", "min_f1", "min_f1_label", "accuracy",
                           "delta_macro_f1", "affected_pct_analysis_set", "pct_full_dataset_affected"]]
                     if len(scen) else pd.DataFrame(), {
                         "macro_f1": ".3f", "min_f1": ".3f", "accuracy": ".3f", "delta_macro_f1": "+.3f",
                         "affected_pct_analysis_set": ".1f", "pct_full_dataset_affected": ".2f"}),
            "- Gộp nhãn gần như **luôn** làm điểm tăng vì bài toán dễ đi; điểm tăng **không** có nghĩa là nên gộp.\n"
            "- Bỏ nhãn nghĩa là mô hình không bao giờ nhận ra loại bài đó, trong khi ngoài đời bài loại đó vẫn xuất hiện.\n"
            "- Tính lại từ dự đoán cũ chỉ là **ước lượng nhanh**; kết quả chính thức cần train lại.\n"
            "- `affected_pct_analysis_set`: % bài bị gộp/bỏ trong tập phân tích (đã cắt mỗi nhãn tối đa "
            f"{s['config']['per_class_cap']} bài); `pct_full_dataset_affected`: % của toàn bộ dataset theo mục 2.\n"]

    out += ["\n## 7. Giới hạn của khảo sát\n"] + [f"- {x}" for x in s.get("limitations", [])]
    out += ["\n## 8. Danh sách file\n"] + [f"- `{f}`" for f in files]
    return "\n".join(out) + "\n"
