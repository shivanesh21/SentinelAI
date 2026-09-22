# SentinelAI — System Architecture & Design

## 1. Overview

SentinelAI is a predictive incident-detection, root-cause-analysis and auto-remediation platform. It ingests operational telemetry (metrics + logs), learns normal system behavior and failure signatures from it, predicts failures before they occur, explains them, and remediates them under safety controls.

This document is the Day 1 deliverable: a working system design derived from the project proposal architecture diagram.

```
              PRODUCTION APPLICATION (simulated services)
                         |
             ------------+-----------
             |                       |
          Metrics                  Logs
             |                       |
             ------------+-----------
                         v
                TELEMETRY PIPELINE
                         v
              Data Processing Layer
                         v
              Feature Engineering
                         v
        --------------------------------
        |       ML INTELLIGENCE        |
        |  Anomaly Detection           |
        |  Failure Prediction          |
        |  Time-Series Forecasting     |
        --------------------------------
                         v
                   Risk Engine
                         v
              Incident Detected
                         v
              LLM Root-Cause Engine
                         v
             Diagnosis + Explanation
                         v
              Remediation Engine  (advisory | approval | autonomous)
                         v
              Recovery Verification
                         v
               MLOps & Drift Monitoring
```

## 2. Component Design

### 2.1 Telemetry Collection (Module 1)
- Simulated multi-service environment (see data strategy doc).
- Collects per-service time-series: CPU, memory, disk, network, request rate, latency, HTTP 4xx/5xx, DB connections, container restarts.
- Collects correlated log lines (INFO/WARNING/ERROR).
- Interval: 30s (config: `config/settings.yaml`).

### 2.2 Feature Engineering (Module 2)
- Rolling-window means/growth rates on windows [5, 10, 15] min.
- Cross-metric ratios: CPU/memory, latency-per-request.
- Labels: `failure_in_next_10min ∈ {0,1}` derived from injected failure ground truth.
- Time-based train/val/test split (no random shuffle — avoids leakage).

### 2.3 Anomaly Detection (Module 3)
- Baselines: z-score / rolling IQR.
- Models: Isolation Forest, Autoencoder (dense/LSTM) trained on normal data only.
- Output: anomaly score mapped to NORMAL / WARNING / ANOMALOUS / CRITICAL.

### 2.4 Failure Prediction (Module 4)
- Candidates: Logistic Regression (baseline), Random Forest, XGBoost, LightGBM, LSTM/Transformer.
- Metrics: F1, Precision, Recall, ROC-AUC, PR-AUC.
- Selected via time-consistent cross-validation; best model(s) persisted under `models/prediction/`.

### 2.5 Risk Scoring (Module 5)
- Weighted fusion of failure probability, anomaly score, error rate, latency trend, memory growth → `incident_risk_score` (0–100%) + severity.

### 2.6 LLM Root-Cause Analysis (Module 6)
- Evidence assembler packages metrics deltas, top anomalous features, correlated log lines.
- LLM (Anthropic/OpenAI/local) prompted with structured evidence; output validated against a Pydantic/JSON schema.
- Output: probable root cause, evidence list, confidence, recommended action.
- Audit trail: every prompt+response persisted.

### 2.7 Remediation (Module 7)
- Action library: restart container, scale replica, rollback, clear cache, notify/ticket.
- Root-cause → action rules.
- Modes: advisory (recommend only), approval (human approves), autonomous (policy-validated, whitelisted safe actions only).

### 2.8 Recovery Verification (Module 8)
- Auto re-check after cooldown; before/after metric comparison; records SUCCESS/FAILURE + recovery time (MTTR).

### 2.9 MLOps (Module 9)
- MLflow experiment tracking, model versioning.
- Drift detection: PSI, KS-test, KL divergence on feature distributions; rolling-F1 model monitoring; retraining trigger.

## 3. Runtime Layout

The repository is a monorepo split into a backend (server) and a frontend (client):

| Layer | Tech |
|---|---|
| Backend package | Python package (`backend/src`) |
| API | FastAPI (`backend/src/api/main.py`) |
| CLI / pipeline scripts | `backend/scripts` |
| Web client (frontend) | Static HTML/CSS/JS served by FastAPI (`frontend/`) |
| Experiment tracking | MLflow (`backend/mlruns/`) |
| Storage | Parquet/CSV partitions under `backend/data/` (no external DB for scope) |
| Deep learning | TensorFlow/Keras |

```
sentinelAI/
  backend/      # server: src/, scripts/, config/, data/, models/, reports/, mlruns/, tests/
  frontend/     # client: static web app (index.html + assets), served at /
  docs/         # platform documentation
```

Run the backend from `backend/`: `python -m uvicorn src.api.main:app --reload --port 8000`.
The single FastAPI process serves both the JSON API (`/api/*`) and the static client at `/`.

## 4. Data Flow (end-to-end path)

`telemetry → raw (data/raw) → features (data/features) → labels (data/labels) → splits (data/splits) → models (predict/anomaly) → risk → rca → remediation → recovery → dashboard`

Orchestrated by `src/pipeline/orchestrator.py` (Day 24 integration point).

## 5. Deployment Model
Single-host demonstration stack:
- Backend package under `backend/src`; scripts in `backend/scripts` for generation/training/pipeline execution (path-independent: they resolve the backend root from their own location).
- The FastAPI app (`backend/src/api/main.py`) exposes `/api/*` and serves the static web client from `frontend/`.
- Services run in the same process; the Docker action wrapper targets the Docker Engine API for remediation demos with a mock fallback so the full flow runs without Docker.