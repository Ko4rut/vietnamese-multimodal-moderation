# Inference API (FastAPI)

Text-only inference over the existing 25-domain **single-label** classifier.
The API does not train models and does not provide image/video moderation.

## Start

Install from repository root:

```bash
python -m pip install -e ".[modeling,api]"
```

Point `VTC_MODEL_DIR` to **one trusted experiment directory** created by
`scripts/06_train_baseline.py`; it must contain `model.joblib`,
`metadata.json`, and `config.json`. Example (replace with your real path):

```bash
export VTC_MODEL_DIR="/path/to/lake/experiments/tfidf_baselines_v2/<experiment-name>"
uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
```

On Windows PowerShell, use
`$env:VTC_MODEL_DIR="G:\\My Drive\\...\\<experiment-name>"`.
A model stored only in Google Drive/Colab must be synced or mounted on the
machine running the API; a path inside a remote Colab runtime is not accessible
by your local computer.

## Endpoints

- `GET /api/health`: process health (does not mean model is loaded).
- `GET /api/status`: readiness, model metadata, domain count, validation Macro-F1.
- `GET /api/labels`: canonical 25-domain label mapping.
- `POST /api/predict`: `{"text":"Văn bản tiếng Việt"}`, length 3–10,000.

The response includes the winning domain, five highest-ranked classes, and
`score_kind`. **Logistic Regression** produces probability estimates;
**LinearSVC** produces uncalibrated decision margins, **not probabilities**.
No prediction is made when the model directory is missing or invalid (HTTP 503).
The server loads the model once per configured directory, so restart after
replacing its files. Never load untrusted `.joblib` artifacts (pickle execution).

Local CORS is limited to Vite's two development origins. For production, serve
the frontend behind a same-origin reverse proxy to `/api`, and add auth/rate
limits, HTTPS, monitoring and appropriate data privacy controls before
public exposure.
