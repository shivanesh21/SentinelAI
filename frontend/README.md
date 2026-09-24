# SentinelAI Frontend

Static web client for the SentinelAI operations dashboard. Plain HTML/CSS/JS with no
build step, served by the FastAPI backend.

## Run

1. Start the backend from the repo root:

   ```powershell
   cd backend
   ..\.venv\Scripts\python -m uvicorn src.api.main:app --reload --port 8000
   ```

   (or activate the conda env `sentinelai` and run `python -m uvicorn src.api.main:app --reload --port 8000`)

2. Open http://localhost:8000 — the FastAPI app serves `frontend/` as static files at `/`
   and exposes the JSON API under `/api`.

## API contract

| Endpoint | Purpose |
|----------|---------|
| `GET /api/health` | Liveness check (`{status, service, time}`) |
| `GET /api/overview` | Dataset summary: rows, services, features, splits |
| `GET /api/risk/current` | Live risk scores + anomaly scores per service |
| `GET /api/rca/history` | Incidents with root-cause explanations (audit trail) |
| `POST /api/rca` | Run RCA on submitted telemetry rows |
| `GET /api/remediate/pending` | Approvals awaiting human sign-off |
| `POST /api/remediate/approve` | Approve + execute a pending remediation |
| `POST /api/remediate` | End-to-end safety-mode remediation (advisory/approval/autonomous) |
| `GET /api/recovery/history` | Recovery verifications (SUCCESS/FAILURE + recovery time) |
| `GET /api/mlops/experiments` | Tracked experiments (Day 21 registry: versions, hyperparams, metrics) |
| `GET /api/mlops/compare` | Best metrics per model family, ranked by metric (`f1`/`roc_auc`/`accuracy`) |
| `GET /api/mlops/drift` | Dataset drift + degradation + retraining decision |

Consumed by `assets/js/dashboard.jsx`; the API base URL defaults to a relative `/api`
(works on any port the backend serves from) and can be overridden via
`window.SENTINEL_API_BASE` before the script loads.

## Layout

```
frontend/
  index.html            # React 18 CDN + Babel shell, mounts dashboard.jsx
  README.md
  assets/
    css/style.css       # dark operations theme
    css/dashboard.css   # dashboard component styles
    js/app.js           # legacy vanilla fetch/render loop
    js/dashboard.jsx    # React dashboard (Overview / Risk / Incidents / Remediation / Models / Drift)
```

The dashboard is a single-page React app loaded from CDNs with no build step:
Overview (health, feature table, service risk + anomaly scores), Risk Scores
(per-service risk table), Incidents (RCA timeline with root-cause explanations),
Remediation (approve/execute pending plans, recovery history), Models (experiment
registry: model comparison + hyperparameters, Day 21), and Drift & ML
(drift report, degradation, retraining trigger).