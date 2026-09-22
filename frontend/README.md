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

| Endpoint          | Purpose                                        |
|-------------------|------------------------------------------------|
| `GET /api/health` | Liveness check (`{status, service, time}`)     |
| `GET /api/overview` | Dataset summary: rows, services, features, splits |

Consumed by `assets/js/app.js`; the API base URL defaults to `http://localhost:8000/api`
and can be overridden via `window.SENTINEL_API_BASE` before the script loads.

## Layout

```
frontend/
  index.html            # page shell + sections
  assets/
    css/style.css       # dark operations theme
    js/app.js           # fetch + render loop (15s refresh)
```

As later modules (prediction, risk, RCA, remediation) ship in the backend, their routes
will drive new sections here: incidents feed, anomaly levels, risk score, and
remediation actions.