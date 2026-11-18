# VietTopic AI — Next.js web demo

A responsive, component-based **Next.js 16 / React 19 / TypeScript / Tailwind CSS 4** UI for the repository's Vietnamese **single-label, 25-domain text classifier**.

The web app calls the existing FastAPI inference service. **No fake AI scores are generated.** If the model or API is unavailable, the page shows its offline state; the example text and reference list of domain labels still work. A staggered animated vertical bar chart renders the API's top 5 labels; Logistic Regression heights use real probabilities while LinearSVC heights use relative ranking and display the original signed decision margins. No mocked prediction scores are generated.

## Run locally

At the repository root (PowerShell):

```powershell
python -m pip install -e ".[modeling,api]"
$env:VTC_MODEL_DIR = "E:\path\to\an\existing\experiment"
python -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
```

The experiment directory **must** contain `model.joblib`, `metadata.json`, and `config.json`. The API is not a trainer. See [API docs](../api/README.md).

In another terminal:

```powershell
cd apps/web
npm ci
npm run dev
```

Open http://localhost:3000. The Next.js server proxies browser requests from `/api/*` to http://127.0.0.1:8000/api/* by default. The browser never needs direct CORS access.

If your API runs at a different address, set **server-only** `API_PROXY_TARGET` before starting Next.js:

```powershell
$env:API_PROXY_TARGET = "http://127.0.0.1:8001"
npm run dev
```

Do not set `NEXT_PUBLIC_` prefixes for the API proxy target. In deployment, the Next.js server must be able to reach the FastAPI server; configure this per environment and secure the API before making it public.

## Organization

```text
src/
├── app/
│   ├── page.tsx                    # Composes the dashboard
│   ├── layout.tsx                  # Metadata and theme bootstrap
│   └── globals.css                 # Light/dark tokens and responsive styling
├── components/
│   ├── layout/{sidebar,topbar}.tsx
│   ├── theme/theme-toggle.tsx
│   └── ui/icon.tsx
└── features/prediction/
    ├── api/client.ts              # Typed HTTP calls (same-origin /api)
    ├── hooks/use-prediction.ts     # Inference state and errors
    ├── data/{examples,domains}.ts
    ├── types.ts
    └── components/
        ├── demo-dashboard.tsx
        ├── text-input-panel.tsx
        ├── prediction-panel.tsx
        ├── prediction-bar-chart.tsx
        ├── prediction-chart.css
        ├── model-status-card.tsx
        └── domain-browser.tsx
```

Theme can be switched manually. The preference is stored in `localStorage` (key: `viettopic-theme`); the first visit uses the device color-scheme preference.

## Quality checks

```powershell
npm run lint
npm run build
```

The web currently supports **text only**. Source-domain labels were automatically assigned in VTSNLP and have not all been independently human-verified; validation metrics should be interpreted with this limitation. The UI sends text to the configured API only when the user requests a prediction; do not submit secrets or sensitive personal data to an untrusted deployment.
