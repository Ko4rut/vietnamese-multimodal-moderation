# VietTopic Web Demo

Responsive React + TypeScript + Vite frontend for the repository's **actual**
25-domain Vietnamese text classification model. This is not a fake scoring
demo: the Predict button is disabled until the API reports a loaded model.

## Start locally

From the repository root, start the API in one terminal (see
[`apps/api/README.md`](../api/README.md)):

```bash
python -m pip install -e ".[modeling,api]"
export VTC_MODEL_DIR="/path/to/lake/experiments/tfidf_baselines_v2/<experiment-name>"
uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd apps/web
npm install
npm run dev
```

Visit **http://localhost:5173**. Vite proxies `/api` to port 8000; no API
URL or secrets need to be embedded in the frontend. Run `npm run build` to
produce static files in `apps/web/dist/`. For deployment configure a reverse
proxy to the API for `/api` on the same origin.

## Display contracts

- Shows model readiness, experiment name and validation Macro-F1 if available.
- Predicts one canonical label per input, with Vietnamese translations.
- Lists top five scores; **probabilities** for Logistic Regression and raw
  **margins** for LinearSVC (never misleading percentages for SVC).
- Accepts text only, max 10,000 characters; no simulated predictions.
- No uploads, local storage, automatic logging or text persistence.

Backend training artifacts are intentionally *not* packaged with the web build.
All example inputs are illustrative; model scores are supplied only by the API.
