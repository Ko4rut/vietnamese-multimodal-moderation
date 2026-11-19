# VietTopic AI — Next.js AI inference dashboard

Responsive **Next.js 16 / React 19 / TypeScript / Tailwind CSS 4** frontend for
Vietnamese single-label text classification across 25 domains. **All predictions
are real FastAPI model inference; there is no mock mode or fabricated result.**

## Run locally

Copy one or more trusted, trained experiment folders into the repository's
`models/` directory. Each needs `model.joblib`, `metadata.json`, and
`config.json`. The backend auto-discovers new experiments.

From the repository root (terminal 1, Python 3.11):

~~~powershell
python -m pip install -e ".[api]"
fastapi dev
~~~

Terminal 2:

~~~powershell
cd apps/web
npm ci
npm run dev
~~~

Visit http://localhost:3000. The browser uses Next.js proxy rewrites from
`/api/*` to http://127.0.0.1:8000/api/*, so it never calls FastAPI directly.
The demo requires an available backend and trusted trained artifact to predict.
The text examples are **sample inputs only**; none have hard-coded predictions.

## Switch models directly in the result panel

Select a baseline experiment in **Trạng thái mô hình**, enter Vietnamese text,
then press **Phân tích văn bản**. When at least two models are registered,
the dashboard selects a pair (preferably Logistic Regression and LinearSVC
from the same V1/V2 generation) and calls `POST /api/compare` **once**.
Both real predictions are cached for that input.

The **Kết quả phân tích** card now has two model tabs above the predicted
label and Top-5 chart. Switch tabs to display the other model's result
**without retyping or rerunning inference**. The Model Status dropdown follows
the active tab. If only one model is present, the existing `POST /api/predict`
flow remains supported; there are no mock predictions.

When more experiments are installed, selecting another experiment from the
dropdown changes the pair. A model that has not yet been analyzed for the
current input is inferred on demand and cached. Editing text or refreshing
the registry clears old predictions so results never belong to an old input.
If a model fails, its tab displays an error rather than fabricating results.

Logistic Regression uses estimated class probabilities; LinearSVC uses
decision margins. Their numeric scales are not directly comparable.

## Compare multiple models

Scroll to **Multi-Model Comparison**, select **2–4** registered models and
press **So sánh model**. The same input text is submitted once to
`POST /api/compare`. Each model gets its own result card, predicted label
and top-five chart; errors are shown individually without hiding successful
results. Results are cleared when the input text or selection changes.

**Do not compare Logistic Regression probabilities numerically to LinearSVC
margins.** Use model predictions/label agreement and class ranks instead.

## API server configuration

To use a backend on another port, set the server-side environment variable
before starting Next.js:

~~~powershell
$env:API_PROXY_TARGET = "http://127.0.0.1:8001"
npm run dev
~~~

Do not prefix that variable with `NEXT_PUBLIC_`. For production, secure both
services (HTTPS, authentication, request limits, monitoring). See
[API documentation](../api/README.md).

## Organization

~~~text
src/
├── app/                          # Layout, theme, route
├── components/                   # Shared layout/UI components
└── features/prediction/
    ├── api/client.ts             # Fully typed FastAPI client
    ├── hooks/
    │   ├── use-prediction.ts     # Paired inference, result caching and readiness
    │   └── use-model-comparison.ts
    ├── utils/model-pair.ts       # Stable V1/V2-aware model selection
    ├── data/                     # Sample input text and canonical labels
    ├── types.ts
    └── components/
        ├── demo-dashboard.tsx
        ├── text-input-panel.tsx
        ├── prediction-panel.tsx
        ├── model-result-tabs.tsx
        ├── model-status-card.tsx
        ├── model-picker.tsx
        ├── prediction-bar-chart.tsx
        ├── model-comparison.tsx
        ├── comparison-result-card.tsx
        └── domain-browser.tsx
~~~

Dark/light mode is supported. Theme preference is stored in `localStorage`
under `viettopic-theme`.

## Quality checks

~~~powershell
npm run lint
npm run test:unit
npm run build
~~~

The application classifies text only (no image/video inference). Dataset
domain labels are source-assigned and have not been fully human-verified.
Only submit text to a backend that you trust.
