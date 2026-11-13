"""Hand-review pack: random samples, confused pairs and suspected mislabels."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .config import SurveyConfig
from .noise import suspected_examples

AUDIT_COLUMNS = [
    "audit_id", "part", "id", "given_label", "model_top1", "model_top1_prob", "model_top2",
    "model_top2_prob", "excerpt", "reviewer", "reviewer_label", "agrees_with_given", "note",
]
_EXCERPT_CHARS = 600


def make_excerpt(text: str, limit: int = _EXCERPT_CHARS) -> str:
    """First ``limit`` characters on one line; neutralise spreadsheet formulas."""
    flat = re.sub(r"\s+", " ", text).strip()[:limit]
    return "'" + flat if flat[:1] in ("=", "+", "-", "@") else flat


def _rows(frame: pd.DataFrame, part: str, texts: Mapping[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "part": part,
            "id": frame["id"].to_numpy(),
            "given_label": frame["given"].to_numpy(),
            "model_top1": frame["top1"].to_numpy(),
            "model_top1_prob": frame["top1_prob"].round(3).to_numpy(),
            "model_top2": frame["top2"].to_numpy(),
            "model_top2_prob": frame["top2_prob"].round(3).to_numpy(),
            "excerpt": [make_excerpt(texts[i]) for i in frame["id"]],
        }
    )


def build_audit_pack(
    oof: pd.DataFrame,
    texts: Mapping[str, str],
    pairs: pd.DataFrame,
    cfg: SurveyConfig,
) -> pd.DataFrame:
    """Parts B (confused pairs) and C (suspects) are chosen first, then part A (random).

    A document appears in at most one part. Rows are shuffled (seeded) so that splitting
    the file by ``audit_id`` hands every reviewer a mix of parts.
    """
    rng = np.random.default_rng(cfg.seed)
    used: set[str] = set()
    parts: list[pd.DataFrame] = []

    # Part B: misread between the two labels of each top pair, both directions.
    for r in pairs.head(cfg.top_pairs).itertuples():
        forward = oof[(oof["given"] == r.label_a) & (oof["top1"] == r.label_b)]
        backward = oof[(oof["given"] == r.label_b) & (oof["top1"] == r.label_a)]
        half = math.ceil(cfg.audit_per_pair / 2)
        pools = [forward[~forward["id"].isin(used)], backward[~backward["id"].isin(used)]]
        want = [half, cfg.audit_per_pair - half]
        picks = []
        for slot, pool in enumerate(pools):
            picks.append(pool.sample(n=min(want[slot], len(pool)), random_state=int(rng.integers(2**31))))
        short = cfg.audit_per_pair - sum(len(p) for p in picks)
        if short > 0:  # one direction is short: fill from the other side
            for slot in (0, 1):
                rest = pools[slot][~pools[slot]["id"].isin(picks[slot]["id"])]
                extra = rest.sample(n=min(short, len(rest)), random_state=int(rng.integers(2**31)))
                picks[slot] = pd.concat([picks[slot], extra])
                short -= len(extra)
        chosen = pd.concat(picks)
        used.update(chosen["id"])
        parts.append(_rows(chosen, "B", texts))

    # Part C: most confident suspected mislabels not already used.
    suspects = suspected_examples(oof, cfg.mismatch_prob)
    suspects = suspects[~suspects["id"].isin(used)].head(cfg.audit_part_c)
    chosen = oof.set_index("id").loc[suspects["id"]].reset_index()
    used.update(chosen["id"])
    parts.append(_rows(chosen, "C", texts))

    # Part A: random documents of every label.
    for label, group in oof.groupby("given", sort=False):
        pool = group[~group["id"].isin(used)]
        chosen = pool.sample(
            n=min(cfg.audit_per_class, len(pool)), random_state=int(rng.integers(2**31))
        )
        used.update(chosen["id"])
        parts.append(_rows(chosen, "A", texts))

    pack = pd.concat([p for p in parts if len(p)], ignore_index=True)
    pack = pack.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    pack.insert(0, "audit_id", [f"AUD-{i:04d}" for i in range(1, len(pack) + 1)])
    for column in ("reviewer", "reviewer_label", "agrees_with_given", "note"):
        pack[column] = ""
    return pack[AUDIT_COLUMNS]


def write_audit_pack(pack: pd.DataFrame, path: Path) -> Path:
    """UTF-8 *with BOM* so Excel and Google Sheets show Vietnamese correctly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pack.to_csv(path, index=False, encoding="utf-8-sig")
    return path


AUDIT_INSTRUCTIONS = """# Hướng dẫn kiểm tra nhãn bằng tay (audit pack)

File `audit_pack.csv` là một bộ bài viết để **người đọc** xem nhãn `domain` do máy gán có hợp lý không.
Mở bằng Google Sheets (File → Import → Upload) hoặc Excel. File đã mã hóa UTF-8 có BOM nên tiếng Việt hiển thị đúng.

> ⚠️ Một số bài thuộc nhãn `Sensitive_Subjects` (và `Adult` nếu có) có thể chứa nội dung nhạy cảm.
> Bạn có thể bỏ qua bài nào không muốn đọc: ghi `unsure` và để trống `reviewer_label`.

## Các cột

| Cột | Ý nghĩa |
|---|---|
| `audit_id` | Số thứ tự. Dùng để chia việc. |
| `part` | `A` = bài ngẫu nhiên của từng nhãn; `B` = bài mô hình đoán nhầm giữa hai nhãn hay lẫn; `C` = bài mô hình rất chắc là nhãn gốc sai. |
| `id` | ID bài trong dataset gốc. |
| `given_label` | Nhãn do máy (nvidia/domain-classifier) gán trong dataset gốc. |
| `model_top1`, `model_top2` (+ xác suất) | Hai nhãn mô hình phân tích đoán. **Chỉ để tham khảo sau khi bạn đã quyết định.** |
| `excerpt` | Khoảng 600 ký tự đầu của bài. |
| `reviewer` | Tên bạn. |
| `reviewer_label` | Nhãn bạn cho là đúng nhất (chọn trong 25 nhãn của repo). Để trống nếu đồng ý với `given_label`. |
| `agrees_with_given` | `yes` / `no` / `unsure` (xem dưới). |
| `note` | Ghi chú tự do, ví dụ "bài quảng cáo", "nửa Sức khỏe nửa Làm đẹp". |

## Cách điền

1. **Đọc `excerpt` trước, tự nghĩ xem bài thuộc nhãn nào, rồi mới nhìn `given_label` và cột `model_*`.** Nhìn nhãn trước dễ làm bạn đồng ý theo.
2. Chỉ cần đọc phần trích; đừng cố đoán phần còn lại của bài.
3. `agrees_with_given`:
   - `yes`: nhãn gốc hợp lý (không nhất thiết là nhãn duy nhất đúng).
   - `no`: có nhãn khác rõ ràng hợp hơn. Điền nhãn đó vào `reviewer_label`.
   - `unsure`: phần trích quá ngắn/rời rạc, bài là rác hay quảng cáo không rõ chủ đề, hai nhãn đều hợp lý như nhau, hoặc bạn không muốn đọc bài. `unsure` là câu trả lời bình thường, không phải thiếu sót; **đừng ép chọn yes/no**.
4. Mỗi người làm độc lập, không hỏi ý nhau trước khi điền.

## Chia bài cho các thành viên

- Các dòng đã được xáo trộn nên chia theo khoảng `audit_id` là đủ công bằng. Ví dụ 4 người: AUD-0001–0100, 0101–0200, ...
- Nên cho khoảng 10–15% số bài được **hai người cùng đọc** (ví dụ khoảng cuối của người kế bên) để biết người đọc có đồng thuận với nhau không. Nếu hai người hay bất đồng thì nhãn đó vốn khó, không phải lỗi của ai.
- Làm xong thì gửi lại file CSV đã điền.
"""
