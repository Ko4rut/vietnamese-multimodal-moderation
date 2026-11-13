# Google Colab notebooks by pipeline version

## V1 — end to end

Open [V1 ingest-to-modeling](v1/colab_end_to_end_v1.ipynb). Execute cells from top to bottom; source code is cloned from `VTC_GIT_REF` (defaults to `main`). It mounts Google Drive, verifies/reuses immutable Bronze/Silver V1/Gold V1 snapshots, then trains **TF-IDF + Logistic Regression** and **TF-IDF + LinearSVC** sequentially. Select on validation Macro-F1; test evaluation is deliberately separate.

Individual V1 stage notebooks: [Bronze](v1/colab_runner.ipynb), [Silver](v1/colab_silver_runner.ipynb), [Gold](v1/colab_gold_runner.ipynb), [Modeling](v1/colab_modeling_runner.ipynb), [Silver EDA](v1/eda/01_silver_eda.ipynb).

## V2 — Silver cleaning

[Silver V2 dedicated runner](v2/colab_silver_v2_runner.ipynb) and [Data Lake with opt-in Silver V2](v2/colab_data_lake_end_to_end.ipynb). The latter builds Gold V1 from an explicitly selected Silver snapshot, but **does not run modeling**. Do not mistake it for the V1 ingest-to-modeling workflow. [Gold V2 runner](v2/colab_gold_v2_runner.ipynb) verifies a published Silver V2 snapshot and optionally creates isolated Gold V2 splits. It does not modify V1 notebooks.

[Modeling V2 runner](v2/colab_modeling_runner.ipynb) retrains TF-IDF + Logistic Regression and LinearSVC from Gold V2, saves to `experiments/tfidf_baselines_v2/`, and reports validation results without touching test.\n\nNotebook versions are organizational, **not** a change to the Drive data layout. Data remains under `01_bronze/`, `02_silver/v1|v2/`, `03_gold/` and `experiments/`. See [Team setup](../docs/COLAB_TEAM_SETUP.md).

## Label survey (EDA, read-only)

[Báo cáo kết quả khảo sát dataset gốc — mẫu 300k, seed 42](reports/label_survey_b81fcce5_seed42_n300000.ipynb) chứa các bảng tổng hợp và 10 biểu đồ nhúng sẵn. Mở xem trực tiếp, không cần chạy lại, cài thư viện hay tải CSV/dataset. Đây là bản kết quả tĩnh của run `survey_b81fcce5_seed42_n300000_69445d1c`; bộ kết quả chi tiết được giữ ngoài Git.

[Label survey (Colab)](v2/eda/02_label_survey_colab.ipynb) khảo sát nhãn `domain` do máy gán của dataset gốc: phân bổ nhãn trên toàn bộ ~12,2 triệu bài, độ đại diện của Bronze 50k, cặp nhãn dễ lẫn (out-of-fold), nghi gán sai, kịch bản gộp/bỏ nhãn "nếu thì" và bộ bài để người đọc kiểm tra tay. **Runtime → Run all** là đủ; kết quả tự đóng zip và tải về. Notebook chỉ tạo bằng chứng, không quyết định nhãn nào giữ/gộp/bỏ và không ghi vào Bronze/Silver/Gold/experiments. Xem [docs/LABEL_SURVEY.md](../docs/LABEL_SURVEY.md). Notebook được sinh bởi `scripts/11_generate_label_survey_notebook.py`.
