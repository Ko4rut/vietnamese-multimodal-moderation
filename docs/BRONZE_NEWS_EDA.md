# EDA, TF-IDF và phân cụm cho nhãn `News` trong Bronze (V3)

Tài liệu cho notebook [`notebooks/v3/eda/01_bronze_news_eda_colab.ipynb`](../notebooks/v3/eda/01_bronze_news_eda_colab.ipynb) và package `vietnamese_topic_classification.analysis.bronze_news`.

## 1. Mục đích và ranh giới

Nhãn `News` (do máy gán) chứa lẫn nhiều chủ đề con nên mô hình khó phân biệt `News` với nhãn khác. Khảo sát nhãn trước đó (`docs/LABEL_SURVEY.md`) ghi nhận cặp Law_and_Government ↔ News bị lẫn nhiều nhất. Notebook này làm rõ "vùng xám" đó trên **Bronze** (khác `news_v3.py`, vốn chạy trên Silver V2), chỉ bằng **TF-IDF** (không embedding, không GPU):

1. Có bao nhiêu bài `News` trong Bronze (đếm lúc chạy, không gán cứng) và dữ liệu sạch cỡ nào?
2. Sau TF-IDF + LSA, dữ liệu có "shape" gì, nên thử thuật toán phân cụm nào?
3. Các thuật toán cho kết quả ra sao, có ổn định và đồng ý với nhau không?
4. Mỗi cụm là chủ đề gì, và **cụm News nào trông giống `Law_and_Government`, `Sports`...?**

**Ranh giới (code và test đảm bảo):**

- **Chỉ đọc** Bronze. Không ghi gì vào Data Lake, kể cả file thử quyền ghi: thư mục kết quả chọn theo thứ tự `OUTPUT_ROOT_OVERRIDE` → `My Drive/vmm_outputs/` → `/content`. Override trỏ vào trong Data Lake bị từ chối.
- **Không quyết định thay nhóm.** Phân cụm không phải gán nhãn; "gợi ý nhãn V3" chỉ là điểm số từ keyword hạt giống, cần người kiểm tra.
- Không có văn bản bài báo trong Git, `eda_summary.json`, `cluster_*.csv/json`, `report.md`. `TextGuard` từ chối ghi nếu một chuỗi ≥ 30 ký tự trùng nguyên văn với một bài. Duy nhất `cluster_excerpts_INTERNAL.csv` có đoạn trích (~500 ký tự): **chỉ dùng nội bộ, không commit** (license dataset chưa xác minh).
- Chạy được trên Colab miễn phí (CPU, ~12 GB RAM), kỳ vọng ~10–15 phút (chưa đo trên Drive thật, xem mục 7).

## 2. Cách chạy

**Colab:** mở
`https://colab.research.google.com/github/Ko4rut/vietnamese-multimodal-moderation/blob/feat/bronze-news-eda-v3/notebooks/v3/eda/01_bronze_news_eda_colab.ipynb`
→ **Runtime → Run all** → chọn tài khoản có Data Lake (quyền Viewer là đủ) rồi Allow. Không có `input()`. Sau khi PR merge, đổi `GIT_REF` thành `"v3_test"` (hoặc `"main"`).

Data Lake được tìm bằng `resolve_mounted_lake` (shortcut `vietnamese-multilabel-topic-classification` trong My Drive, hoặc `DATA_LAKE_ROOT_OVERRIDE`). Manifest Bronze tự chọn bản `_seed42_…_n50000.json` (`find_bronze_manifest`), hoặc đặt `BRONZE_MANIFEST_NAME`.

**Local / test:**

```bash
python -m pip install -e ".[dev,survey]"
pytest -q tests/test_bronze_news_eda.py tests/test_bronze_news_eda_notebook.py
python scripts/16_generate_bronze_news_eda_notebook.py --check
```

Test chạy offline trên Data Lake giả (manifest + JSONL Bronze nhỏ, sha256 đúng, văn bản tổng hợp). Chính các cell của notebook (trừ clone/pip/mount Drive) được chạy trên lake giả. Test tự bỏ qua khi thiếu pandas/scikit-learn/IPython.

## 3. Quy trình

| Mục | Làm gì | Module |
|---|---|---|
| 1 | Verify Bronze (`BronzeReader.verify`), lọc `domain == "News"`, bảng phân bố mọi nhãn | `loading.py` |
| 2 | Chất lượng (rỗng, `id` trùng, trùng sha256, near-duplicate), độ dài, nhiễu crawl, từ vựng, so với nhãn láng giềng | `eda.py` |
| 3 | Chuẩn hóa, TF-IDF (1–2 gram), LSA (TruncatedSVD + L2) | `features.py` |
| 4 | Hopkins, k-distance, phân bố cosine, elbow/silhouette, BIC/AIC, PCA + t-SNE, diễn giải tự động | `shape.py` |
| 5 | Registry thuật toán, quét k, metric, độ ổn định, đồng thuận, chọn cấu hình chính | `clustering.py` |
| 6 | Từ khóa cụm, bài đại diện, gợi ý nhãn V3, vùng xám với nhãn láng giềng | `interpret.py`, `seed_keywords.yaml` |
| 7 | Báo cáo markdown, xuất file, chống ghi đè, zip | `report.py`, `output.py`, `plots.py` |

### Đặc trưng

- Chuẩn hóa dùng lại `SilverTransformer.normalize_text` (NFC + khoảng trắng), rồi chữ thường, bỏ URL/email; token là âm tiết (`_TOKEN` của `news_v3.py`, chỉ chữ cái nên số và dấu câu tự mất); stopword = `_STOPWORDS` của `news_v3.py` mở rộng thêm từ chức năng tiếng Việt.
- TF-IDF: `ngram_range=(1,2)`, `sublinear_tf=True` như baseline (`training/factory.py`). **Khác baseline:** `min_df=3` (baseline 2) và `max_df=0.6`, `max_features=30000`, vì ~3000 bài nên từ xuất hiện 2 lần quá hiếm để làm tâm cụm.
- Giảm chiều: `TruncatedSVD` 100 chiều rồi chuẩn hóa L2, nên khoảng cách Euclid ≈ cosine. Lý do: TF-IDF có hàng chục nghìn cột gần như toàn 0, mọi cặp bài đều "xa nhau như nhau", các thuật toán dựa trên khoảng cách chạy kém.
- Bài rỗng và bản sao trùng chính xác (khi `DROP_EXACT_DUPLICATES=True`) bị bỏ khỏi bước phân cụm; số bài bỏ được in ra. EDA vẫn báo cáo đủ và **không xóa** gì khỏi dữ liệu nguồn.

### Chẩn đoán shape và chọn thuật toán

Thuật toán **không chọn sẵn**: `plan_algorithms` quyết định chạy/bỏ dựa trên số đo, và mọi quyết định đều có lý do in ra (và vào `eda_summary.json`).

- **Hopkins** (lấy min của hai bản: cột bị xáo trộn độc lập ở toàn bộ chiều, và đồng đều trong hộp bao ở 10 chiều đầu). Hopkins ở chiều cao thường bị thổi phồng, chỉ là dấu hiệu.
- **k-distance** (k=10): gợi ý `eps` của DBSCAN (điểm gập), và độ đều mật độ (p95/p50, tỉ lệ điểm lẻ).
- Phân bố cosine giữa các cặp (độ tập trung), silhouette theo k (KMeans), BIC/AIC của GMM (`diag`).
- Ngưỡng diễn giải là heuristic, được in kèm theo (`shape.THRESHOLDS`).
- Quy tắc bỏ qua hiện có: `meanshift` bị bỏ khi dữ liệu ≥ 30 chiều (KDE ở chiều cao mất ý nghĩa). Thuật toán tắt trong `ALGORITHMS` được ghi "tắt trong cấu hình". Thuật toán chậm (OPTICS ≤ 1500 bài, Spectral ≤ 2000, AffinityPropagation ≤ 1000) chạy trên mẫu theo seed rồi gán phần còn lại bằng láng giềng gần nhất, kèm ngân sách thời gian `TIME_BUDGET_SECONDS` (kiểm tra giữa các cấu hình, không ngắt giữa chừng một lần fit).

### Chọn cấu hình chính

Điểm tổng hợp = trung bình thứ hạng phần trăm của silhouette (cosine), Davies–Bouldin (thấp tốt), Calinski–Harabasz. Cấu hình chính phải: 5–15 cụm (`main_min_clusters=5`, vì k quá nhỏ gộp nhiều chủ đề, không dùng để khảo sát chủ đề con), nhiễu ≤ 30%, cụm nhỏ nhất ≥ 1%, ổn định giữa các seed (ARI trung bình ≥ 0.5, thuật toán xác định tính là 1.0), và 2 cấu hình phải thuộc 2 thuật toán khác nhau. Không có cấu hình nào đạt thì lấy cấu hình xếp hạng cao nhất và ghi rõ. Đồng thuận giữa các thuật toán đo bằng ARI/NMI.

### Vùng xám

Mẫu tối đa `NEIGHBOR_PER_LABEL=600` bài mỗi nhãn láng giềng (reservoir, theo seed) được **chiếu vào không gian TF-IDF/SVD đã fit trên News** (không fit lại, cụm News nguyên vẹn; từ chỉ có ở nhãn láng giềng bị bỏ qua). Với mỗi cụm:

- `nearest_label_by_centroid`: nhãn láng giềng có tâm gần tâm cụm nhất (cosine).
- `knn_share_<nhãn>` và `leak_rate`: mỗi bài News bỏ phiếu bằng 15 láng giềng gần nhất trong tập gộp (News + mẫu láng giềng), phiếu được chia cho kích thước nhãn để nhãn lớn không thắng chỉ nhờ số lượng. `leak_rate` = tỉ lệ bài News mà đa số phiếu thuộc nhãn khác. **Đây là độ giống về dùng từ, không phải bằng chứng gán sai.**

### Gợi ý nhãn V3

`seed_keywords.yaml` (sửa tự do; test kiểm tra tên nhãn thuộc `domains_v3.yaml`) liệt kê âm tiết/cụm 2 âm tiết cho `News_Politics`, `News_Crime_Disaster`, `News_Current_Affairs`, `Sports`, `Law_and_Government`, `Business_and_Industrial`, `Finance`, `Health`, `People_and_Society`, `Sensitive_Subjects`. Điểm = phần trọng số từ khóa của cụm khớp với hạt giống nhãn đó; từ khóa 2 âm tiết khớp nếu có đúng cụm đó, hoặc (nửa trọng số) cả hai âm tiết cùng nằm trong top từ khóa. Danh sách hạt giống là bản nháp của người viết code, **chưa được người kiểm duyệt ngôn ngữ rà**.

## 4. Cấu hình

Mọi tham số nằm ở cell `config` (`BronzeNewsConfig.from_mapping` đọc thẳng biến của notebook). Các tham số chính:

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `GIT_REF` | `feat/bronze-news-eda-v3` | Nhánh clone trên Colab |
| `SEED` | 42 | Seed của mọi bước ngẫu nhiên |
| `USE_DRIVE` / `DATA_LAKE_ROOT_OVERRIDE` | `True` / `None` | Mount Drive hay dùng thư mục có sẵn |
| `BRONZE_MANIFEST_NAME` | `None` | Tự tìm manifest 50k seed 42 |
| `OUTPUT_ROOT_OVERRIDE` | `None` | Ép nơi lưu (không được nằm trong Data Lake) |
| `INCLUDE_NEIGHBOR_LABELS`, `NEIGHBOR_LABELS`, `NEIGHBOR_PER_LABEL` | `True`, 5 nhãn, 600 | Vùng xám |
| `DROP_EXACT_DUPLICATES` | `True` | Bỏ bản sao trùng khi phân cụm |
| `MIN_DF`, `MAX_DF`, `MAX_FEATURES`, `NGRAM_RANGE`, `SVD_COMPONENTS` | 3, 0.6, 30000, (1,2), 100 | TF-IDF và LSA |
| `K_RANGE` | (3, 15) | Quét k |
| `ALGORITHMS` | tất cả `True` | Bật/tắt từng thuật toán |
| `SLOW_MAX_POINTS`, `TIME_BUDGET_SECONDS` | 1500, 300 | Giới hạn thuật toán chậm |
| `EXPORT_RESULTS`, `AUTO_DOWNLOAD_ZIP` | `True` | Ghi file, tự tải zip |

## 5. Kết quả

`<OUTPUT_ROOT>/bronze_news_eda/<run_id>/` (+ `<run_id>.zip` cạnh đó), `run_id = bronze_news_<sha8 Bronze>_seed<seed>_<hash8 cấu hình>`:

| File | Nội dung |
|---|---|
| `cluster_assignments.csv` | `id`, cụm của từng cấu hình chính, tọa độ PCA/t-SNE, khoảng cách tới tâm. Không có text |
| `cluster_summary.csv` / `.json` | Kích thước, từ khóa, gợi ý nhãn V3 + điểm, nhãn láng giềng gần nhất, độ dài |
| `algorithm_comparison.csv` | Metric của mọi cấu hình, thời gian, trạng thái (kể cả cấu hình bị bỏ qua và lý do) |
| `eda_summary.json` | Số liệu EDA, shape, kế hoạch thuật toán, cấu hình chính. Không có text |
| `figures/*.png` | Biểu đồ chính |
| `report.md` | Tóm tắt tự động (chỉ số liệu) |
| `cluster_excerpts_INTERNAL.csv` | Đoạn trích bài đại diện. **Chỉ dùng nội bộ** |
| `run_config.json`, `_COMPLETE.json` | Cấu hình; dấu hoàn tất (ghi cuối cùng) |

**Không ghi đè:** thư mục đã có `_COMPLETE.json` thì chạy lại chỉ đóng zip lại. Thư mục dở dang (không có `_COMPLETE.json`) được ghi lại, từng file ghi qua file tạm rồi đổi tên.

## 6. Giới hạn

- TF-IDF chỉ thấy mặt chữ: cùng chủ đề khác từ ngữ có thể tách cụm, khác chủ đề dùng nhiều từ chung có thể gộp cụm. Phân cụm không phải gán nhãn.
- Nhãn `domain` do máy gán, nên cả "nhãn láng giềng" cũng có nhiễu.
- Không gian fit trên News: bài láng giềng chỉ được chiếu vào; độ giống là ước lượng.
- Hopkins ở chiều cao bị thổi phồng; ngưỡng diễn giải, tiêu chí chọn cấu hình chính và quy tắc bỏ qua thuật toán là heuristic.
- Dữ liệu là mẫu Bronze 50k (seed 42), không phải toàn bộ dataset.
- Mọi số liệu thật (số bài News, shape, cụm...) chỉ có sau khi chạy trên Drive; tài liệu này không đưa số liệu thật.

## 7. Chưa kiểm chứng

Tác giả không truy cập được Drive thật. Test dùng dữ liệu giả, nên **chưa biết**: thời gian chạy thật của ~3000 bài (ước lượng 10–15 phút), chất lượng bộ keyword hạt giống trên văn bản thật, và hành vi của Hopkins/ngưỡng shape trên dữ liệu thật. Cần xem lại sau lần chạy đầu tiên.
