# Khảo sát nhãn `domain` của VTSNLP/vietnamese_curated_dataset

Tài liệu cho notebook [`notebooks/v2/eda/02_label_survey_colab.ipynb`](../notebooks/v2/eda/02_label_survey_colab.ipynb) và package `vietnamese_topic_classification.analysis.label_survey`.

## 1. Mục đích và ranh giới

Cột `domain` của dataset gốc do **máy** gán (`nvidia/domain-classifier`, DeBERTa-v3-base đọc tối đa 512 token), không phải người. Trước khi nhóm quyết định giữ, gộp hay bỏ nhãn nào, cần bằng chứng. Khảo sát này trả lời:

1. Trên **toàn bộ** dataset (~12,2 triệu bài), mỗi nhãn chiếm bao nhiêu? Có nhãn lạ, rỗng, hay `Adult` không?
2. Mẫu Bronze 50k đang dùng có **đại diện** cho toàn bộ dataset không?
3. Mỗi nhãn "trông như thế nào": độ dài, từ đặc trưng?
4. **Cặp nhãn nào dễ lẫn**, lẫn tới mức nào (đo bằng nhiều cách độc lập)?
5. Nhãn nào có dấu hiệu **gán sai** nhiều?
6. Gộp/bỏ nhãn thì điểm số thay đổi thế nào (kịch bản "nếu thì", chỉ để tham khảo)?
7. Một **bộ bài cho người đọc kiểm tra tay** (audit pack).

**Ranh giới (đã được code và test đảm bảo):**

- Chỉ tạo bằng chứng và kịch bản; **không** quyết định nhãn nào giữ/gộp/bỏ. Báo cáo chỉ nêu sự thật và con số.
- **Không ghi** vào Bronze, Silver, Gold hay `experiments/`, không sửa `config/domains_v1.yaml`. Mọi thứ ghi vào `04_reports/label_survey/`.
- **Không đụng tập test của Gold.** Mẫu khảo sát loại **toàn bộ ID có trong Bronze 50k** (Silver/Gold đều là tập con của Bronze), nên không trùng bất kỳ dữ liệu train/validation/test nào của nhóm.
- Không đưa văn bản bài viết vào Git (dataset chưa rõ license), không đưa vào `summary.json`/báo cáo. `audit_pack.csv` có đoạn trích ~600 ký tự và **chỉ để chia sẻ nội bộ**, không commit.
- Không cần GPU, token Hugging Face hay API trả phí.

## 2. Cách chạy

**Colab (khuyến nghị):** mở
`https://colab.research.google.com/github/Ko4rut/vietnamese-multimodal-moderation/blob/feat/label-survey/notebooks/v2/eda/02_label_survey_colab.ipynb`
→ **Runtime → Run all** → khi Colab hỏi quyền Google Drive, chọn đúng tài khoản có Data Lake rồi Allow. Không có `input()`, không cần sửa tham số. Để nguyên tab; xong thì file `label_survey_<run_id>.zip` tự tải về, một bản nằm ở `<DATA_LAKE_ROOT>/04_reports/label_survey/`. Sau khi PR được merge vào `main`, đổi `GIT_REF` trong cell cấu hình thành `"main"`.

Chạy được trên Colab miễn phí (CPU, ~12 GB RAM). Nếu Drive không mount được hoặc `USE_DRIVE=False`, notebook vẫn chạy ở chế độ không Drive: lưu vào `/content/label_survey_outputs/`, **bỏ qua** phần so sánh với Bronze 50k (ghi rõ "bỏ qua" trong báo cáo, không coi như bằng nhau) và không loại được ID Bronze khỏi mẫu.

**Chạy lại:** cùng cấu hình thì bỏ qua phần đã xong (cache). Thư mục kết quả có `_COMPLETE.json` thì **không bao giờ bị ghi đè**, chỉ đóng zip lại. Chưa hoàn tất thì chạy tiếp các bước còn thiếu.

**Local / test:** `python -m pip install -e ".[dev,survey]"`, `pytest -q tests/test_label_survey_*.py`. Test chạy offline bằng parquet giả; chính các cell của notebook (trừ clone/pip/mount Drive) được chạy trên dữ liệu giả trong `tests/test_label_survey_notebook.py`.

## 3. Cấu hình (cell cấu hình của notebook)

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `GIT_REF` | `feat/label-survey` | Nhánh code được clone trên Colab |
| `HF_REVISION` | `b81fcce5…` | Revision dataset được ghim (như `domains_v1.yaml`) |
| `SEED` | 42 | Seed cho mọi bước ngẫu nhiên |
| `USE_DRIVE` | `True` | `False`: chỉ lưu `/content`, bỏ qua so sánh Bronze |
| `BRONZE_MANIFEST_NAME` | `None` | `None`: tự tìm bản 50k/seed 42 và in ra bản đã chọn |
| `SURVEY_TARGET_ROWS` | 300 000 | Số bài của mẫu khảo sát |
| `MAX_DOWNLOAD_GB` | 4.0 | Trần dung lượng tải (tính theo kích thước nén của cột id/text/domain) |
| `PER_CLASS_CAP` | 4 000 | Mỗi nhãn tối đa bao nhiêu bài khi train mô hình phân tích |
| `TEXT_CHAR_LIMIT` | 5 000 | Cắt bài khi đưa vào TF-IDF |
| `HEAD_CHARS` | 1 000 | Thí nghiệm "chỉ đọc đầu bài" |
| `N_FOLDS` | 5 | Số lượt out-of-fold |
| `TOP_PAIRS` | 15 | Số cặp lẫn nhiều nhất để xem kỹ |
| `MISMATCH_PROB` | 0.70 | Ngưỡng "mô hình rất chắc là nhãn khác" |
| `AUDIT_PER_CLASS` / `AUDIT_PER_PAIR` / `AUDIT_PART_C` | 12 / 10 / 30 | Kích thước audit pack |
| `RUN_HEAD_VS_FULL` | `True` | Chạy thí nghiệm đầu bài |
| `AUTO_DOWNLOAD_ZIP` | `True` | Tự tải zip về máy |

Run id: `survey_<rev8>_seed<SEED>_n<rows>_<confighash8>`. `confighash8` là hash của mọi tham số ảnh hưởng kết quả (không gồm số luồng) và SHA-256 của Bronze, nên đổi tham số hoặc đổi Bronze ra một lần chạy mới, không đè lên lần cũ.

## 4. Kiểm kê repo: dùng lại / sửa nhẹ / viết mới

| Việc | Quyết định | Lý do |
|---|---|---|
| Cấu hình TF-IDF + Logistic Regression | **Dùng lại** `training.factory.build_pipeline` (lấy vectorizer/classifier từ pipeline) | Giữ đúng baseline (ngram 1–2, `min_df=2`, `max_features=100000`) |
| Accuracy, Macro-F1, F1 từng nhãn | **Dùng lại** `evaluation.metrics.calculate_metrics` | Cùng định nghĩa metric với baseline |
| Chuẩn hóa văn bản | **Dùng lại** `SilverTransformer.normalize_text` | Cùng quy tắc NFC/khoảng trắng của Silver V1 |
| Đọc Bronze/Silver/Gold | **Dùng lại** `BronzeReader`, `SilverReader`, `GoldReader` (chỉ đọc, có `verify()`) | Có kiểm tra checksum |
| Tìm Data Lake trên Drive | **Dùng lại** `colab_lake.resolve_mounted_lake` | Thông báo lỗi tiếng Việt rõ ràng |
| Khuôn cell đầu (clone theo ref, `pip install -e`, refresh import) | **Dùng lại khuôn** `notebooks/v2/colab_modeling_runner.ipynb` | Xử lý module cũ giữa các lần checkout trên Colab |
| Style biểu đồ, cách sinh notebook + test `--check` | **Dùng lại** `COLORS` của `01_silver_eda.ipynb`, mô hình `scripts/04_…` + `tests/test_silver_eda_notebook.py` | Đồng bộ với repo |
| `evaluation/diagnostics.py` (confusion matrix, F1) | **Không dùng được**: gắn chặt với artifact Gold và thư mục ghi một lần | Viết hàm vẽ mới nhỏ (`confusion.py`), không đụng API/test cũ |
| Đọc parquet từ Hugging Face, đếm cột `domain`, lấy mẫu row group | **Viết mới** (`hf_parquet.py`) | Chưa có |
| Cặp lẫn, dendrogram, nghi gán sai, kịch bản, audit pack, báo cáo, runner | **Viết mới** | Chưa có |

Không sửa file cũ nào ngoài: `pyproject.toml` (thêm extra `survey`) và hai `README` (thêm mục). `ruff check src tests` và toàn bộ test cũ vẫn xanh.

**Việc còn lại cho CI (cần quyền `workflow`):** để CI chạy các test `test_label_survey_*`, đổi bước cài đặt trong `.github/workflows/ci.yml` từ `.[dev,modeling]` thành `.[dev,modeling,survey]`. Khi chưa đổi, các test này tự `skip` (không làm CI đỏ) vì thiếu pandas/pyarrow.

## 5. Các bước và file kết quả

Logic nằm trong package; notebook chỉ gọi `SurveyRun.step_*`. Mỗi bước in thời gian ước tính trước và thời gian thật sau, và ghi dấu hoàn tất trong `steps/`.

| Bước (mục notebook) | File chính |
|---|---|
| 4. Phân bổ toàn bộ (§5.1) | `dataset_metadata.csv`, `full_label_counts.csv`, `file_position_label_pct.csv`, `label_position_drift.csv`, `fig_full_label_distribution.png`, `fig_label_by_file_position.png` |
| 5. Bronze có đại diện không (§5.2) | `bronze_vs_full.csv`, `snapshot_vs_full.csv` (Silver/Gold train, nếu đọc được), `fig_bronze_vs_full.png` |
| 6. Mẫu khảo sát ngẫu nhiên (§5.3) | `survey_sample_label_counts.csv`, `survey_sample_vs_full.csv`, `fig_survey_sample_vs_full.png` (mẫu parquet nằm trong cache, không nằm trong zip) |
| 7. Hồ sơ nhãn (§5.4) | `length_profile.csv`, `fig_length_by_label.png`, `top_terms_by_label.csv`, `conflicting_duplicates.csv` (chỉ ID + nhãn) |
| 8. Lẫn lộn giữa các nhãn (§5.5) | `confusion_matrix_oof.csv`, `confusing_pairs.csv`, `per_label_metrics_oof.csv`, `head_vs_full.csv`, `analysis_set_counts.csv`, `fig_confusion_matrix_oof.png`, `fig_top_confusing_pairs.png`, `fig_label_dendrogram.png`, `fig_per_label_f1.png` |
| 9. Nghi gán sai (§5.6) | `suspected_mislabels_by_label.csv`, `suspected_mislabels_ids.csv` |
| 10. Kịch bản và audit (§5.7–5.8) | `label_scenarios.csv`, `greedy_merge_curve.csv`, `fig_greedy_merge_curve.png`, `audit_pack.csv`, `audit_instructions.md` |
| 11. Báo cáo | `summary.json` (không chứa văn bản bài), `label_survey_report.md`, `_COMPLETE.json`, zip |

### Cách đọc các số đo lẫn

- `sym_confusion = (C[a,b] + C[b,a]) / 2`, `C` là ma trận lẫn out-of-fold chuẩn hóa theo hàng (nhãn gốc a bị đoán thành b).
- `top2_ambiguity`: trong các bài có nhãn gốc là a hoặc b, tỉ lệ bài có {top-1, top-2} đúng bằng {a, b} và hai xác suất cách nhau dưới 0,2.
- `centroid_cosine`: cosine giữa vector TF-IDF trung bình của a và của b (không dùng mô hình phân loại).
- Dendrogram: khoảng cách = 1 − `sym_confusion`/max, `linkage(method="average")`; thứ tự lá cũng là thứ tự hàng/cột của `confusion_matrix_oof.csv`.
- Ba tín hiệu độc lập; cặp xuất hiện cao ở cả ba thì bằng chứng chắc hơn.

### Kịch bản "nếu thì"

S0 giữ nguyên; S1 gộp riêng từng cặp trong `TOP_PAIRS`; S2 gộp tham lam 1→8 bước, **tính lại ma trận lẫn sau mỗi lần gộp**; S3 bỏ từng nhãn trong 5 nhãn F1 thấp nhất cùng `News`, `Sensitive_Subjects` (bài bị đoán thành nhãn đã bỏ thì dùng top-2). Mỗi dòng có `affected_pct_analysis_set` (trên tập phân tích đã cắt mỗi nhãn tối đa `PER_CLASS_CAP`) và `pct_full_dataset_affected` (theo phân bổ toàn bộ ở bước 4). Gộp nhãn gần như luôn làm điểm tăng; điểm tăng **không** có nghĩa là nên gộp. Bỏ nhãn nghĩa là mô hình không bao giờ nhận ra loại bài đó. Tính lại từ dự đoán cũ chỉ là ước lượng nhanh; kết quả chính thức cần train lại.

## 6. Quyết định thiết kế đáng biết

**Đọc parquet bằng range request song song, không dùng `HfFileSystem.open`.** Dataset gồm 132 file train (~260 MB mỗi file), mỗi file 93 row group × 1.000 bài. Cột `domain` chỉ ~1 KB mỗi row group nên đếm nhãn không cần tải văn bản. Nhưng đo thực tế cho thấy `HfFileSystem.open` mất ~0,5–1 s cho mỗi range (mỗi lần lại qua redirect Hub → CDN): ~50–100 s/file, tức ~2 giờ tuần tự cho 132 file. `hf_parquet.py` vẫn dùng `HfFileSystem` để liệt kê file ở revision đã ghim, nhưng đọc dữ liệu bằng một bộ đọc range keep-alive chỉ dùng thư viện chuẩn (resolve redirect một lần, tự resolve lại khi URL ký hết hạn, từ chối tải toàn file nếu server bỏ qua `Range`). Footer parquet được đọc bằng một request 512 KB cuối file.

**Chiến lược lấy mẫu chọn từ metadata.** Row group chỉ có 1.000 bài liên tiếp nên lấy cả row group sẽ cho ~300 cụm bài nằm cạnh nhau. Thay vào đó: xáo tất cả (file, row group) bằng `SEED`, lấy nhiều row group nhất mà trần `MAX_DOWNLOAD_GB` cho phép, và giữ một tập con ngẫu nhiên dòng trong mỗi row group (seed theo `(seed, file, row group)`), nhờ đó mẫu trải ra hàng nghìn vị trí. Chiến lược thực tế (số row group, số dòng mỗi row group, dung lượng dự kiến) được ghi trong `summary.json`.

**TF-IDF fit một lần.** Vectorizer được fit một lần trên toàn bộ tập phân tích (không dùng nhãn) thay vì mỗi fold, để tiết kiệm thời gian trên Colab; bộ phân loại thì luôn dự đoán bài chưa từng thấy.

## 7. Dữ liệu thật và thời gian chạy

Các số dưới đây đo bằng một lần chạy thật trên Hugging Face (revision đã ghim) trong quá trình phát triển, **không phải** trên Colab và **không** có Drive (nên chưa so được với Bronze 50k).

**Metadata của dataset (đọc từ footer parquet):**

- 132 file `train-xxxxx-of-00132.parquet`, mỗi file 92 190–92 191 dòng, 247–276 MB (tổng 34,6 GB).
- 12 276 row group, **mỗi row group đúng 1 000 dòng** (93 row group mỗi file).
- Tổng **12 169 131 bài**. Dung lượng nén của cột `text` ≈ 34,5 GB (≈ 2,8 MB mỗi row group), cột `id` ≈ 83 MB, cột `domain` ≈ 14 MB (≈ 107 KB mỗi file).
- Đếm nhãn trên toàn bộ chỉ chuyển ~83 MB qua 12 276 request nhỏ (gồm cửa sổ 512 KB cuối mỗi file để đọc footer), tức văn bản không bao giờ được tải.

**Phân bổ nhãn toàn bộ** (đúng 25 giá trị, **không** có `Adult`, null hay rỗng; nhãn lớn nhất / nhỏ nhất ≈ 18,0 lần):

| # | Nhãn | Số bài | % |
|---|---|---|---|
| 1 | `Arts_and_Entertainment` | 941,159 | 7.73% |
| 2 | `Health` | 886,657 | 7.29% |
| 3 | `People_and_Society` | 843,624 | 6.93% |
| 4 | `Sports` | 793,235 | 6.52% |
| 5 | `News` | 769,510 | 6.32% |
| 6 | `Sensitive_Subjects` | 743,582 | 6.11% |
| 7 | `Food_and_Drink` | 675,447 | 5.55% |
| 8 | `Business_and_Industrial` | 624,991 | 5.14% |
| 9 | `Travel_and_Transportation` | 581,803 | 4.78% |
| 10 | `Jobs_and_Education` | 564,654 | 4.64% |
| 11 | `Beauty_and_Fitness` | 555,714 | 4.57% |
| 12 | `Home_and_Garden` | 529,616 | 4.35% |
| 13 | `Computers_and_Electronics` | 516,543 | 4.24% |
| 14 | `Books_and_Literature` | 396,770 | 3.26% |
| 15 | `Internet_and_Telecom` | 379,943 | 3.12% |
| 16 | `Law_and_Government` | 358,097 | 2.94% |
| 17 | `Games` | 345,954 | 2.84% |
| 18 | `Autos_and_Vehicles` | 328,610 | 2.70% |
| 19 | `Real_Estate` | 319,040 | 2.62% |
| 20 | `Finance` | 304,233 | 2.50% |
| 21 | `Pets_and_Animals` | 189,001 | 1.55% |
| 22 | `Shopping` | 187,516 | 1.54% |
| 23 | `Science` | 182,682 | 1.50% |
| 24 | `Hobbies_and_Leisure` | 98,387 | 0.81% |
| 25 | `Online_Communities` | 52,363 | 0.43% |

**Nhãn có theo vị trí file không?** Phân bổ khá đều giữa các shard nhưng không hoàn toàn: nhãn trôi nhiều nhất là `Sensitive_Subjects` (từ 5,0% tới 8,3% giữa các shard, tương quan −0,65 với vị trí file), kế đó `Home_and_Garden`, `News`, `Books_and_Literature` (độ lệch chuẩn ≤ 0,2 điểm %). TVD của riêng shard 0 so với toàn bộ là 0,020 (lệch lớn nhất 1,06 điểm % ở `Sensitive_Subjects`), của 5 shard đầu là 0,0125, trung vị của từng shard là 0,0096. Bronze 50k chỉ gồm chưa đầy hai shard (mỗi shard ~92 000 dòng; bộ xáo streaming `buffer_size=10_000` chỉ trộn trong một bộ đệm nhỏ, còn thứ tự shard được `datasets` xáo theo seed), nên Bronze có thể lệch cỡ độ lệch của một shard: TVD của từng shard so với toàn bộ có trung vị 0,0096 và lớn nhất 0,048. Con số chính thức cho Bronze cần Drive (bước 5 của notebook).

**Đường chạy đầu-cuối trên Hugging Face thật** (cấu hình tí hon: 4 row group, 2 000 bài, `N_FOLDS=2`): xong trong ~16 s sau khi có cache đếm nhãn.

**Benchmark thời gian/bộ nhớ** (mẫu 75 000 bài, 344 row group ≈ 1,0 GB, 23 904 bài trong tập phân tích, `N_FOLDS=5`, `n_jobs=2`, máy dev 12 nhân; **chưa đo trên Colab**):

| Bước | Thời gian |
|---|---|
| Đếm nhãn 132 file (mạng có độ trễ cao, 24 luồng) | ~20 phút (chạy một lần, rồi cache) |
| Tải 1,0 GB mẫu | 88 s |
| TF-IDF + 5 fold Logistic Regression (đầy đủ) | 123 s, mỗi fold 31–42 s, không cảnh báo hội tụ |
| Thí nghiệm đầu bài | 73 s |
| RSS đỉnh | ≈ 2,5 GB |

Ngoại suy tuyến tính cho cấu hình mặc định (300 000 bài, tập phân tích ≈ 90 000 bài, ~3,8 lần lớn hơn): mẫu ≈ 3,1 GB; mô hình đầy đủ ≈ 8 phút và đầu bài ≈ 5 phút trên máy dev; Colab miễn phí (2 vCPU, chậm hơn khoảng 2–2,5 lần) khoảng 30–50 phút cả notebook, bộ nhớ đỉnh dự kiến khoảng 6–8 GB trong 12 GB. Đây là **ước lượng, chưa kiểm chứng trên Colab**. Nếu cần nhanh hơn, hạ `PER_CLASS_CAP` (ví dụ 2 000) hoặc `TEXT_CHAR_LIMIT`, hoặc tắt `RUN_HEAD_VS_FULL`.

## 8. Giới hạn của khảo sát

- Nhãn do máy gán, chưa người kiểm duyệt; "nghi gán sai" chỉ là nghi vấn.
- Mô hình phân tích (TF-IDF + Logistic Regression, out-of-fold) là công cụ đo, không phải mô hình chính thức; không so trực tiếp với baseline Gold (khác dữ liệu, khác cách cân bằng nhãn).
- Mỗi nhãn bị cắt tối đa `PER_CLASS_CAP` bài nên tỉ lệ trong tập phân tích không phản ánh ngoài đời.
- Chỉ chuẩn hóa mức Silver V1 (NFC, khoảng trắng); **không** làm sạch Silver V2, nên nhiễu crawl có thể làm tăng độ lẫn.
- Bài dài bị cắt (`TEXT_CHAR_LIMIT`); bộ gán nhãn gốc chỉ đọc ~512 token đầu (thí nghiệm "đầu bài" kiểm tra điều này).
- Kích thước mẫu hiệu dụng có thể nhỏ hơn số dòng nếu các bài trong cùng một row group giống nhau.
- Kịch bản gộp/bỏ nhãn tính lại từ dự đoán cũ, chỉ là ước lượng nhanh.
