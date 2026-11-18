# VietTopic AI — Next.js web demo

A responsive, component-based **Next.js 16 / React 19 / TypeScript / Tailwind CSS 4** UI for the repository's Vietnamese **single-label, 25-domain text classifier**.

The web app defaults to **Mock Preview**: four fully typed, static example predictions demonstrate the animated Top-5 bar chart without requiring FastAPI. These values are fabricated UI fixtures, **not** trained-model predictions. Select **AI thực** to call the existing FastAPI inference service instead. Logistic Regression scores use a 0–100% probability scale, while LinearSVC uses relative bar heights with original decision margins shown as numbers.

## Mock Preview (no backend needed)

From the repository root:

```powershell
cd apps/web
npm ci
npm run dev
```

Open http://localhost:3000. The first view immediately shows a **MOCK DATA** Top-5 chart. Choose one of the four exact example texts (Công nghệ, Tài chính, Thể thao, Du lịch) to switch between typed fixtures; the button can replay the column animation. Custom text is **never** assigned a pretend classification: choose a fixed example or select **AI thực** for real inference.

## Run locally with real model

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
    ├── data/{examples,domains,mock-predictions}.ts
    ├── types.ts
    └── components/
        ├── demo-dashboard.tsx
        ├── text-input-panel.tsx
        ├── prediction-mode-switch.tsx
        ├── prediction-panel.tsx
        ├── mock-preview.css
        ├── prediction-bar-chart.tsx
        ├── prediction-chart.css
        ├── model-picker.tsx
        ├── model-picker.css
        ├── model-status-card.tsx
        └── domain-browser.tsx
```

Theme can be switched manually. The preference is stored in `localStorage` (key: `viettopic-theme`); the first visit uses the device color-scheme preference.

## Quality checks

```powershell
npm run lint
npm run build
```

Mock Preview never posts example text to the API; switching to **AI thực** enables genuine inference and its status checks. The web currently supports **text only**. Source-domain labels were automatically assigned in VTSNLP and have not all been independently human-verified; validation metrics should be interpreted with this limitation. The UI sends text to the configured API only when the user requests a prediction; do not submit secrets or sensitive personal data to an untrusted deployment.
