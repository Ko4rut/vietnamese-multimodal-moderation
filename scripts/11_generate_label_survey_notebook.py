"""Generate the Colab label-survey notebook (standard library only).

Edit the cells here, then run ``python scripts/11_generate_label_survey_notebook.py``.
``--check`` fails when ``notebooks/v2/eda/02_label_survey_colab.ipynb`` is out of date.
Cells tagged ``colab-only`` need Colab (git clone, pip, Drive mount); the offline test
executes every other cell against fake data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1] / "notebooks/v2/eda/02_label_survey_colab.ipynb"
)

INTRO = """# Khảo sát nhãn `domain` của VTSNLP/vietnamese_curated_dataset

**Chỉ cần: Runtime → Run all**, rồi bấm cho phép khi Colab hỏi quyền Google Drive. Không cần nhập gì, không cần sửa tham số.

## Notebook làm gì?
Cột `domain` (25 nhãn như `Health`, `News`, `Sports`...) do **máy** gán, không phải người. Trước khi quyết định giữ, gộp hay bỏ nhãn nào, nhóm cần **bằng chứng**. Notebook này thu thập bằng chứng đó và trả lời:

1. Trên **toàn bộ** dataset (~12,2 triệu bài), mỗi nhãn chiếm bao nhiêu? Có nhãn lạ hay nhãn rỗng không?
2. Mẫu Bronze 50k nhóm đang dùng có **đại diện** cho toàn bộ dataset không?
3. Mỗi nhãn "trông như thế nào" (độ dài bài, từ đặc trưng)?
4. **Cặp nhãn nào dễ lẫn nhau**, lẫn tới mức nào?
5. Nhãn nào có dấu hiệu **gán sai** nhiều?
6. Nếu gộp hoặc bỏ một số nhãn thì điểm số đổi thế nào? (chỉ để tham khảo)
7. Một **bộ bài để người đọc kiểm tra tay**.

**Notebook không tự quyết định** nhãn nào giữ, gộp hay bỏ. Nó chỉ tạo số liệu; quyết định do người đưa ra sau.

## Chạy bao lâu, kết quả ở đâu?
- Khoảng **30–60 phút** trên Colab miễn phí (CPU, ~12 GB RAM). Mỗi bước nặng in thời gian ước tính trước và thời gian thật sau khi chạy.
- Chạy lại cùng cấu hình thì **bỏ qua phần đã xong** (cache trên Drive), không làm lại từ đầu.
- Cuối cùng file `label_survey_<...>.zip` **tự tải về máy**; một bản nằm trên Drive trong `04_reports/label_survey/`.
- Không cần GPU, không cần token Hugging Face. Không ghi vào Bronze/Silver/Gold hiện có."""

CONFIG = '''# ===== Tất cả tham số nằm ở đây. Mặc định đã ổn: không cần sửa gì rồi chạy tiếp. =====
GIT_REF = "feat/label-survey"   # sau khi merge vào main thì đổi thành "main"
HF_DATASET = "VTSNLP/vietnamese_curated_dataset"
HF_REVISION = "b81fcce58945970117a1b56d50ec81be2628a5c3"   # phiên bản dataset được ghim cố định
SEED = 42

USE_DRIVE = True                # False: chỉ lưu vào /content, bỏ qua phần so sánh với Bronze 50k
BRONZE_MANIFEST_NAME = None     # None: tự tìm (ưu tiên bản 50k, seed 42) và in ra bản đã chọn

SURVEY_TARGET_ROWS = 300_000    # số bài muốn lấy cho mẫu khảo sát ngẫu nhiên
MAX_DOWNLOAD_GB = 4.0           # trần dung lượng tải, tới trần thì dừng lấy thêm
PER_CLASS_CAP = 4_000           # mỗi nhãn tối đa bao nhiêu bài khi train mô hình phân tích
TEXT_CHAR_LIMIT = 5_000         # cắt bài quá dài khi đưa vào TF-IDF
HEAD_CHARS = 1_000              # thí nghiệm "chỉ đọc đầu bài"
N_FOLDS = 5                     # số lượt out-of-fold
TOP_PAIRS = 15                  # số cặp nhãn lẫn nhiều nhất để xem kỹ
MISMATCH_PROB = 0.70            # mô hình đoán nhãn khác với xác suất >= ngưỡng này thì bị nghi gán sai
AUDIT_PER_CLASS = 12            # audit pack: số bài ngẫu nhiên mỗi nhãn
AUDIT_PER_PAIR = 10             # audit pack: số bài đoán nhầm mỗi cặp

RUN_HEAD_VS_FULL = True         # thí nghiệm chỉ đọc HEAD_CHARS ký tự đầu bài (tốn thêm vài phút)
AUTO_DOWNLOAD_ZIP = True        # tự tải file zip về máy khi xong

# --- Nâng cao (thường không cần đổi) ---
AUDIT_PART_C = 30               # audit pack: số bài nghi gán sai nhất
ROWS_PER_ROW_GROUP = None       # None: tự chọn từ metadata parquet
MAX_FEATURES = 100_000          # kích thước từ điển TF-IDF (như baseline của repo)
LR_MAX_ITER = 300               # số vòng lặp tối đa của Logistic Regression
HTTP_WORKERS = 24               # số kết nối tải song song
N_JOBS = 2                      # số tiến trình train song song (Colab miễn phí có 2 CPU)

print("Cấu hình đã nạp. Nhánh code:", GIT_REF, "| seed:", SEED)'''

CLONE = '''from pathlib import Path
import subprocess

REPO_URL = "https://github.com/Ko4rut/vietnamese-multimodal-moderation.git"
PROJECT_DIR = Path("/content/vmm_label_survey_project")

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
            f"\\"main\\" ở cell cấu hình. Chi tiết: {result.stderr.strip()}"
        )

revision = subprocess.check_output(
    ["git", "-C", str(PROJECT_DIR), "rev-parse", "--short", "HEAD"], text=True
).strip()
print(f"Repository: {GIT_REF} @ {revision}")'''

INSTALL = '''import importlib
import subprocess
import sys

# Chỉ cài các thư viện phân tích nhẹ (không có torch / transformers).
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

from vietnamese_topic_classification.config import get_domain_labels
assert len(get_domain_labels()) == 25
print("Python:", sys.version.split()[0], "| số nhãn chuẩn:", len(get_domain_labels()))'''

DRIVE = '''from pathlib import Path

DRIVE_OK = False
DATA_LAKE_ROOT = None
if USE_DRIVE:
    try:
        from google.colab import drive
        from vietnamese_topic_classification.data.colab_lake import resolve_mounted_lake

        MOUNT = "/content/vtsnlp_drive"
        drive.mount(MOUNT)   # Colab sẽ hỏi quyền: chọn đúng tài khoản có Data Lake rồi bấm Allow
        DATA_LAKE_ROOT = resolve_mounted_lake(MOUNT)
        DRIVE_OK = True
        print("Data Lake:", DATA_LAKE_ROOT)
    except Exception as exc:
        print("⚠️ Không dùng được Google Drive, chạy tiếp ở chế độ KHÔNG Drive "
              "(bỏ qua so sánh với Bronze 50k, kết quả lưu ở /content).")
        print(f"   Lý do: {type(exc).__name__}: {exc}")
else:
    print("USE_DRIVE=False: chạy không Drive (bỏ qua so sánh với Bronze 50k).")

OUTPUT_ROOT = (
    DATA_LAKE_ROOT / "04_reports" / "label_survey" if DRIVE_OK
    else Path("/content/label_survey_outputs")
)
print("Thư mục kết quả:", OUTPUT_ROOT)
if not DRIVE_OK:
    print("⚠️ /content bị xóa khi Colab ngắt kết nối: hãy tải file zip về ngay khi xong.")'''

BRONZE = '''from vietnamese_topic_classification.analysis.label_survey.lake_inputs import (
    load_bronze, snapshot_label_counts,
)

bronze, snapshots = None, {}
BRONZE_SKIP_REASON = "USE_DRIVE=False hoặc không mount được Google Drive"
if DRIVE_OK:
    try:
        bronze = load_bronze(DATA_LAKE_ROOT, BRONZE_MANIFEST_NAME)   # đọc + kiểm tra checksum, chỉ đọc
        print(f"Bronze: {bronze.records:,} bài, {len(bronze.ids):,} ID sẽ bị loại khỏi mẫu khảo sát.")
        snapshots = snapshot_label_counts(DATA_LAKE_ROOT)             # Silver/Gold train, chỉ đọc
    except Exception as exc:
        bronze, snapshots = None, {}
        BRONZE_SKIP_REASON = f"không đọc được Bronze: {type(exc).__name__}: {exc}"
        print("⚠️ Bỏ qua phần so sánh với Bronze 50k. Lý do:", BRONZE_SKIP_REASON)
        print("   Lưu ý: mẫu khảo sát sẽ KHÔNG loại được ID của Bronze (khoảng 0,4% bài có thể trùng).")'''

BUILD = '''from vietnamese_topic_classification.analysis.label_survey.config import SurveyConfig
from vietnamese_topic_classification.analysis.label_survey.runner import SurveyRun
from IPython.display import display

cfg = SurveyConfig.from_mapping(globals())
run = SurveyRun(
    cfg, OUTPUT_ROOT, bronze=bronze, snapshots=snapshots,
    bronze_skip_reason=BRONZE_SKIP_REASON, repo_dir=PROJECT_DIR,
)
print("run_id:", run.run_id)
print("Kết quả:", run.out_dir)
if run.complete:
    print("✔ Lần chạy này đã hoàn tất trước đó: các bước sẽ chỉ hiển thị lại kết quả, không ghi đè.")'''

SECTIONS = [
    (
        "## 4. Phân bổ nhãn trên TOÀN BỘ dataset\n\n"
        "**Để làm gì:** đếm mỗi nhãn trên cả ~12,2 triệu bài mà **không tải văn bản** (chỉ đọc cột `domain`, "
        "khoảng 1 KB mỗi nhóm 1.000 bài), kèm bảng metadata của dataset.\n\n"
        "**Đọc kết quả:** `pct` là % bài của nhãn. Kiểm tra bảng có dòng `unknown`/`null_value`/`empty_value` "
        "hay nhãn `Adult` không. Heatmap theo vị trí file cho biết dữ liệu có bị xếp theo nhãn/nguồn không; "
        "nếu có, mẫu lấy từ vài file đầu (như Bronze) sẽ bị lệch.",
        "run.step_full_counts()\n"
        "display(run.table('dataset_metadata').describe().round(1))\n"
        "display(run.table('full_label_counts'))\n"
        "display(run.table('label_position_drift').head(8))\n"
        "run.show_figures('fig_full_label_distribution.png', 'fig_label_by_file_position.png')",
    ),
    (
        "## 5. Mẫu Bronze 50k có đại diện cho toàn bộ không?\n\n"
        "**Để làm gì:** so phân bổ nhãn của Bronze 50k (và Silver/Gold nếu đọc được) với toàn bộ dataset. "
        "Bronze lấy bằng bộ đệm xáo 10.000 dòng nên thực chất đến từ vài file đầu, không phải mẫu ngẫu nhiên.\n\n"
        "**Đọc kết quả:** `diff_pct_points` là chênh lệch tính bằng điểm %. TVD (total variation distance) "
        "là tổng độ chênh chia đôi: 0 là giống hệt, 1 là hoàn toàn khác. Kiểm định chi-square với n lớn gần như "
        "luôn cho p rất nhỏ, nên **nhìn độ chênh là chính**. Nếu không có Drive, phần này được ghi rõ là \"bỏ qua\".",
        "run.step_bronze_compare()\n"
        "print(run.summary['bronze_vs_full'])\n"
        "if run.summary['bronze_vs_full'].get('status') == 'ok':\n"
        "    display(run.table('bronze_vs_full'))\n"
        "    run.show_figures('fig_bronze_vs_full.png')\n"
        "    if (run.out_dir / 'snapshot_vs_full.csv').is_file():\n"
        "        display(run.table('snapshot_vs_full'))",
    ),
    (
        "## 6. Mẫu khảo sát ngẫu nhiên trên toàn bộ dataset\n\n"
        "**Để làm gì:** lấy một mẫu thật sự ngẫu nhiên (khác cách của Bronze): chọn ngẫu nhiên các nhóm 1.000 bài "
        "trên mọi file, rồi giữ một phần bài ngẫu nhiên trong mỗi nhóm. **Loại toàn bộ ID có trong Bronze 50k**. "
        "Chuẩn hóa văn bản tối thiểu (NFC, như Silver V1); **không** áp dụng làm sạch Silver V2.\n\n"
        "**Đọc kết quả:** nếu mẫu đại diện tốt thì TVD so với toàn bộ rất nhỏ (cỡ 0,01 hoặc thấp hơn). "
        "Bước này tải nhiều dữ liệu nhất; đã tải thì lần sau đọc lại từ cache.",
        "run.step_sample()\n"
        "info = run.summary['survey_sample']\n"
        "print('Chiến lược:', info['strategy'])\n"
        "print(f\"Số bài: {info['rows_final']:,} | loại vì trùng ID Bronze: {info['excluded_bronze_ids']:,} | \"\n"
        "      f\"TVD so với toàn bộ: {info['tvd_vs_full']:.4f}\")\n"
        "display(run.table('survey_sample_vs_full'))\n"
        "run.show_figures('fig_survey_sample_vs_full.png')",
    ),
    (
        "## 7. Hồ sơ từng nhãn\n\n"
        "**Để làm gì:** độ dài bài theo nhãn, tỉ lệ bài dài hơn khoảng 256 và 512 tiếng (giúp quyết định cắt bài cho "
        "PhoBERT sau này), và các nhóm bài **giống hệt nhau nhưng khác nhãn**.\n\n"
        "**Đọc kết quả:** \"tiếng\" tách theo khoảng trắng, **không phải token** của mô hình. Nhóm trùng khác nhãn "
        "là bằng chứng trực tiếp rằng cùng một nội dung bị gán nhãn khác nhau.",
        "run.step_profile()\n"
        "display(run.table('length_profile').round(2))\n"
        "print(run.summary['conflicting_duplicates'])\n"
        "run.show_figures('fig_length_by_label.png')",
    ),
    (
        "## 8. Bằng chứng lẫn lộn giữa các nhãn (phần quan trọng nhất)\n\n"
        "**Để làm gì:** train một mô hình đơn giản (TF-IDF + Logistic Regression, cấu hình như baseline của repo) "
        "nhưng mỗi bài được dự đoán bởi mô hình **chưa từng thấy bài đó** (out-of-fold, chia 5 phần). "
        "Mỗi nhãn chỉ lấy tối đa `PER_CLASS_CAP` bài để nhãn lớn không lấn át nhãn nhỏ; nhờ đó độ lẫn phản ánh "
        "**độ giống nhau về nội dung** chứ không phải độ đông của nhãn. Đây là mô hình **để phân tích**, "
        "không phải mô hình chính thức, nên không so điểm của nó với baseline Gold.\n\n"
        "**Đọc kết quả:** `sym_confusion` càng cao thì hai nhãn càng hay bị nhầm với nhau; `top2_ambiguity` là "
        "tỉ lệ bài mà hai nhãn đó gần ngang xác suất; `centroid_cosine` đo độ giống của \"bài trung bình\" hai nhãn. "
        "Ba cách đo độc lập: nếu cùng chỉ một cặp thì bằng chứng chắc hơn. Dendrogram cho thấy nhãn nào tự nhiên "
        "đi cùng nhau. Bước này lâu nhất (vài lần fit Logistic Regression).",
        "run.step_model()\n"
        "run.step_head_vs_full()\n"
        "run.step_confusion()\n"
        "print('Chỉ số tổng (out-of-fold):', {k: round(v, 3) for k, v in run.summary['model'].items() "
        "if isinstance(v, float)})\n"
        "display(run.table('confusing_pairs').head(cfg.top_pairs).round(3))\n"
        "display(run.table('per_label_metrics_oof').sort_values('f1').round(3))\n"
        "if run.summary.get('head_vs_full'):\n"
        "    print('Chỉ đọc đầu bài:', run.summary['head_vs_full'])\n"
        "    display(run.table('head_vs_full').round(3))\n"
        "run.show_figures('fig_confusion_matrix_oof.png', 'fig_top_confusing_pairs.png',\n"
        "                 'fig_label_dendrogram.png', 'fig_per_label_f1.png')",
    ),
    (
        "## 9. Dấu hiệu nhãn bị gán sai\n\n"
        "**Để làm gì:** đếm bài mà mô hình out-of-fold đoán nhãn **khác** với xác suất ≥ `MISMATCH_PROB`, "
        "và nhãn đó hay bị đọc thành nhãn nào.\n\n"
        "**Đọc kết quả:** đây là **nghi vấn**, không phải kết luận: mô hình cũng có thể sai. "
        "Hãy đối chiếu với bộ audit (mục 10) để người đọc xác nhận.",
        "run.step_noise()\n"
        "display(run.table('suspected_mislabels_by_label').round(3))",
    ),
    (
        "## 10. Kịch bản \"nếu thì\" và bộ bài kiểm tra tay\n\n"
        "**Để làm gì:** tính nhanh (không train lại) điểm số nếu gộp một cặp, gộp tham lam từng bước, hoặc bỏ một nhãn; "
        "và tạo `audit_pack.csv` cho người đọc.\n\n"
        "**Đọc kết quả:** gộp nhãn gần như **luôn** làm điểm tăng vì bài toán dễ đi, điểm tăng **không** có nghĩa là "
        "nên gộp. Bỏ nhãn nghĩa là mô hình không bao giờ nhận ra loại bài đó trong khi ngoài đời bài đó vẫn xuất hiện. "
        "Con số tính lại từ dự đoán cũ chỉ là ước lượng nhanh. Cột `pct_full_dataset_affected` cho biết bao nhiêu % "
        "toàn bộ dataset bị ảnh hưởng.",
        "run.step_scenarios()\n"
        "run.step_audit()\n"
        "display(run.table('label_scenarios').round(3))\n"
        "print('Audit pack:', run.summary['audit_pack'])\n"
        "run.show_figures('fig_greedy_merge_curve.png')",
    ),
]

FINALIZE = '''zip_path = run.finalize()   # viết summary.json + báo cáo, đánh dấu hoàn tất, đóng zip
print("Báo cáo:", run.out_dir / "label_survey_report.md")
print("File zip:", zip_path, f"({zip_path.stat().st_size / 1e6:.1f} MB)")
print("Thời gian từng bước (giây):", run.timings)

from vietnamese_topic_classification.analysis.label_survey.report import try_download
if AUTO_DOWNLOAD_ZIP:
    try_download(zip_path)   # nếu trình duyệt chặn tải, hàm in hướng dẫn tải tay'''

HANDOFF = '''print("=" * 70)
print("HOÀN TẤT. Gửi lại cho người phân tích: cả file zip, hoặc tối thiểu các file sau:")
for name in ("label_survey_report.md", "summary.json", "confusing_pairs.csv", "label_scenarios.csv",
             "per_label_metrics_oof.csv", "full_label_counts.csv"):
    print("  -", name)
print("  - và các ảnh fig_*.png")
print("\\nThư mục kết quả:", run.out_dir)
print("Bản zip:", zip_path)
print("\\nCòn một việc cho cả nhóm: điền audit_pack.csv (xem audit_instructions.md) rồi gửi lại.")'''


def cells() -> list[tuple[str, str, str, list[str]]]:
    """(kind, id, source, tags)."""
    out: list[tuple[str, str, str, list[str]]] = [
        ("markdown", "intro", INTRO, []),
        ("markdown", "config-md", "## 1. Cấu hình\n\nTất cả tham số nằm ở một cell, có chú thích. **Không cần sửa** để chạy bình thường.", []),
        ("code", "config", CONFIG, []),
        ("markdown", "install-md", "## 2. Cài đặt\n\nClone repo theo `GIT_REF`, cài các thư viện phân tích nhẹ (`pip install -e \".[survey]\"`) và nạp lại module để Colab dùng đúng phiên bản code.", []),
        ("code", "clone", CLONE, ["colab-only"]),
        ("code", "install", INSTALL, ["colab-only"]),
        ("markdown", "drive-md", "## 3. Google Drive và thư mục kết quả\n\nColab sẽ hỏi quyền Google Drive: chọn **đúng tài khoản có Data Lake** rồi bấm Allow. Nếu mount lỗi hoặc `USE_DRIVE=False`, notebook vẫn chạy tiếp ở chế độ không Drive (kèm cảnh báo, và bỏ qua phần so sánh với Bronze 50k).\n\nKết quả ghi vào `04_reports/label_survey/<run_id>/`; cache nằm ở `04_reports/label_survey/_cache/`. **Không ghi đè** Bronze/Silver/Gold/experiments.", []),
        ("code", "drive", DRIVE, ["colab-only"]),
        ("code", "bronze", BRONZE, ["colab-only"]),
        ("code", "build-run", BUILD, []),
    ]
    for index, (markdown, code) in enumerate(SECTIONS):
        number = 4 + index
        out.append(("markdown", f"section-{number}-md", markdown, []))
        out.append(("code", f"section-{number}", code, []))
    out += [
        ("markdown", "report-md", "## 11. Báo cáo và file zip\n\nViết `summary.json` và `label_survey_report.md` (tiếng Việt, chỉ nêu sự thật và con số, **không** khuyến nghị giữ/gộp/bỏ nhãn), đóng zip và tự tải về. Nếu trình duyệt chặn tải, notebook in đường dẫn trên Drive và cách tải tay (bảng Files bên trái → chuột phải → Download).", []),
        ("code", "finalize", FINALIZE, []),
        ("markdown", "handoff-md", "## 12. Gửi kết quả cho người phân tích", []),
        ("code", "handoff", HANDOFF, []),
    ]
    return out


def build_notebook() -> dict:
    notebook_cells = []
    for kind, cell_id, source, tags in cells():
        cell = {
            "cell_type": kind,
            "id": cell_id,
            "metadata": {"tags": tags} if tags else {},
            "source": source,
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        notebook_cells.append(cell)
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"name": "02_label_survey_colab.ipynb", "provenance": []},
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
