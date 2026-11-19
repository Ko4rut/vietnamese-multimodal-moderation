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

## Windows PowerShell: run locally (no model configuration required)

Copy a **trusted** trained experiment folder from Google Drive into the
repository's `models/` directory. Each experiment keeps its original
`model.joblib`, `metadata.json`, `config.json` (and optional report files).

~~~text
models/
|-- tfidf_linear_svc_v2_9ff85ff3a7/
|   |-- model.joblib
|   |-- metadata.json
|   |-- config.json
|   `-- validation_metrics.json
`-- tfidf_logreg_v2_9ff85ff3a7/
    |-- model.joblib
    |-- metadata.json
    `-- config.json
~~~

At the repository root, activate your Python 3.11 environment and run:

~~~powershell
python -m pip install -e ".[api]"
fastapi dev
~~~

Open http://127.0.0.1:8000/docs and check `GET /api/models`. New model
folders show up on the next request without restarting FastAPI. The registry
reads metadata only; `GET /api/status` and `POST /api/predict` load the selected
trusted joblib model on demand. Replacing files in an **already loaded**
experiment still requires a process restart because the loader caches it.
Never unpickle models from untrusted sources.

### Optional custom model storage

`VTC_MODELS_ROOT` overrides the project-local `models/` directory. Use it for
a synced Drive folder, a shared server, or a mounted Docker volume:

~~~powershell
./scripts/run_inference_api.ps1 -ModelsRoot "E:/models/experiments" -DefaultModel "tfidf_linear_svc_v2_9ff85ff3a7"
~~~

The backward-compatible single-experiment option also works:

~~~powershell
./scripts/run_inference_api.ps1 -ModelDir "E:/models/experiments/tfidf_baselines_v2/tfidf_logreg_v2_9ff85ff3a7"
~~~

If neither root nor single-model override is set, the repository's `models/`
directory is used by default. `VTC_DEFAULT_MODEL` is optional; without it,
the first registered model ID alphabetically becomes the default. Incorrect
explicit paths or an unknown configured default produce a service error rather
than silently selecting the wrong model.

`fastapi dev` uses auto-reload and localhost by default. The PowerShell helper
also runs this CLI. For production/container usage, use `fastapi run`.
The command must be launched with the application installed; the FastAPI
entrypoint is configured in `pyproject.toml`.

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
