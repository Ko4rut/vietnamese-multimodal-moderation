# FastAPI inference service

This service loads trusted TF-IDF + Logistic Regression / LinearSVC artifacts
previously trained by the project. It does **not** train or download models.
Do not upload or commit model.joblib files to GitHub.

## Verified artifact layout (Google Drive)

~~~text
experiments/
├── tfidf_baselines_v1/
│   ├── tfidf_logreg_v1_3b778f51b8/
│   └── tfidf_linear_svc_v1_3b778f51b8/
└── tfidf_baselines_v2/
    ├── tfidf_logreg_v2_9ff85ff3a7/
    └── tfidf_linear_svc_v2_9ff85ff3a7/
~~~

Each model folder contains model.joblib, config.json and metadata.json.
The four real artifacts were trained with **scikit-learn 1.6.1**. The
`api` dependency extra pins the compatible version when loading these joblib artifacts. Download
or sync experiment folders onto the host running the API; a path in a Colab
runtime cannot be read from your Windows PC.

## Windows PowerShell: run locally

From the repository root:

~~~powershell
python -m pip install -e ".[api]"

# Replace with the actual local path of your synced Drive experiments.
.\scripts\run_inference_api.ps1 -ModelsRoot "E:\models\experiments" -DefaultModel "tfidf_linear_svc_v2_9ff85ff3a7"
~~~

For a single experiment, use the backward-compatible option:

~~~powershell
.\scripts\run_inference_api.ps1 -ModelDir "E:\models\experiments\tfidf_baselines_v2\tfidf_logreg_v2_9ff85ff3a7"
~~~

Run `fastapi dev` from the **repository root**; the app entrypoint is set in
`pyproject.toml`, so no module path is needed:

~~~powershell
$env:VTC_MODELS_ROOT = "E:\\models\\experiments"
$env:VTC_DEFAULT_MODEL = "tfidf_linear_svc_v2_9ff85ff3a7"
fastapi dev
~~~

`fastapi dev` uses auto-reload and localhost by default. The PowerShell helper
also runs this CLI. For production/container usage, use `fastapi run` instead.
Visit http://127.0.0.1:8000/docs (interactive Swagger UI).

## Docker Compose: local container

Docker Desktop must be running. From the repository root:

~~~powershell
$env:VTC_MODELS_HOST_PATH = "E:\models\experiments"
$env:VTC_DEFAULT_MODEL = "tfidf_linear_svc_v2_9ff85ff3a7"
docker compose -f apps/api/compose.yml up --build
~~~

The mount is read-only. Port 8000 is bound only to localhost.

## Backend architecture

~~~text
apps/api/
├── main.py                   # create_app(), include_router(), CORS
├── route/
│   ├── health.py             # GET health, labels
│   ├── model.py              # GET models, status
│   └── prediction.py         # POST predict, HTTP errors
├── schema/
│   ├── health.py             # HealthStatus
│   ├── model.py              # ModelCatalog, ModelStatus
│   └── prediction.py         # PredictRequest, PredictResponse
├── service/
│   ├── types.py              # ModelDescriptor, LoadedModel, errors
│   ├── registry.py           # Discover/validate experiment metadata
│   ├── loader.py             # Read/cache trusted joblib files
│   └── inference.py          # Infer and rank model scores
├── Dockerfile
└── compose.yml
~~~

All user-facing API paths and JSON response fields are preserved; moving
implementation code does not require changes to the Next.js client.

## API contract

| Method | Route | Purpose |
|---|---|---|
| GET | /api/health | Process health, not model readiness |
| GET | /api/models | Registered model IDs and public metadata |
| GET | /api/status?model_id=... | Readiness of selected model |
| GET | /api/labels | Canonical 25 domains |
| POST | /api/predict | Actual model inference, ranked top 5 |

~~~powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/models"

$body = @{ text = "Thị trường chứng khoán tăng điểm."; model_id = "tfidf_linear_svc_v2_9ff85ff3a7" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/predict" -Method Post -ContentType "application/json; charset=utf-8" -Body $body
~~~

The request model_id is optional; defaults to VTC_DEFAULT_MODEL or the first
registered model. Never pass a file path from the browser. The response includes
model_id, model_name, label_id, label, score_kind, top_predictions. Logistic
Regression emits probability estimates, while LinearSVC emits raw decision
margins; these are **not** percentages. Invalid text returns HTTP 422,
unknown model_id 404, and unavailable/incompatible artifacts 503.

The Next.js web app already proxies /api/* to localhost:8000. Start the web
app and choose **AI thực** rather than Mock Preview to test real inference.

Before any public Internet deployment, add HTTPS, authentication, request
limits, monitoring and a trusted persistent artifact volume.
