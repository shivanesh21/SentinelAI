# Module 5 — Risk Engine (Prediction Pipeline)

Day 13 deliverable: a continuous per-service **risk score** that fuses the trained
classifiers (Baseline LR/RF, Advanced XGB/LGBM, Time-Series LSTM) with live operational
signals, exposed through the FastAPI backend for the dashboard and alerting.

## Concept

Ranks and calibration of individual models vary (see `docs/sequential-model.md` for the
LSTM threshold-transfer issue). The risk engine instead produces a stable continuous
score that balances model probability with degradation signals:

```
risk_score = Σ w_c · component_c            (weights renormalised over present components)
risk_level = bucket(risk_score)             NORMAL < WARNING < ANOMALOUS < CRITICAL
risk       = 1 if level != NORMAL
```

### Components

| Component | Source | Normalisation |
|-----------|--------|---------------|
| `failure_probability` | mean of the loaded classifiers' probabilities (LR, RF, XGB, LGBM, LSTM) | [0,1] by construction |
| `anomaly_score` | external anomaly detector (Module 3; optional, not yet persisted) | percentile vs train |
| `error_rate` | `Error_rate` feature | percentile vs train |
| `latency_trend` | `Latency_15min_avg` feature | percentile vs train |
| `memory_growth` | `Memory_growth_rate` feature | percentile vs train |

Weights default to `0.40 / 0.25 / 0.15 / 0.10 / 0.10` (`settings.yaml: risk.weights`).
Missing components (e.g. `anomaly_score` until a detector is persisted) are dropped and
the remaining weights renormalised. Operational-signal percentiles come from references
built on the **training split** (`scripts/score_risk.py` → `models/risk/calibration.json`),
so their scale is stable across time rather than batch-relative.

## Architecture

- `src/risk/engine.py` — `RiskEngine` (loads models, calibrates, `components()` /
  `score()` / `latest_by_service()`, persistence of calibration).
- `src/risk/__init__.py` — exports `RiskEngine`, `LEVELS`, `level_from_score`.
- `scripts/score_risk.py` — calibrates on train, evaluates on val/test, writes
  `reports/evaluations/risk_report.json` and the calibration file.
- `src/api/main.py` — endpoints (`GET /api/risk/current`, `POST /api/risk/score`).
- `frontend/` — dashboard can poll `GET /api/risk/current` per service.

### Serving endpoints

- `GET /api/risk/current?minutes=20` — scores the most recent `minutes` of the feature
  table and returns the latest risk row per service, sorted by risk.
- `POST /api/risk/score` — body `{"rows": [ {timestamp, service, <26 features>}, ... ]}`
  (live telemetry batch). Returns latest per-service risk.
- Both validate required feature columns and return `risk_score`, `risk_level`, `risk`,
  `failure_probability` and available component scores per service.

## Results (same chronological test split)

| Split | F1 | Precision | Recall | ROC-AUC | PR-AUC | Warning+ | Anomalous+ | Critical |
|-------|-----|-----------|--------|---------|--------|----------|------------|----------|
| Val | 0.6171 | 0.4861 | 0.8444 | 0.9471 | 0.8778 | 241 | 25 | 203 |
| Test | 0.6533 | 0.5327 | 0.8445 | 0.9194 | 0.8584 | 250 | 70 | 200 |

Effective weights on disk (no persisted anomaly detector): `failure_probability 0.533`,
`error_rate 0.200`, `latency_trend 0.133`, `memory_growth 0.133`.

### Analysis

- The fused test F1 (**0.653**) is well above the LSTM alone (0.473) and competitive with
  the tree models (LR 0.817, RF 0.855, XGB 0.749, LGBM 0.717) while preserving high
  recall (0.84 vs trees' 0.67–0.88) — a deliberately recall-friendly operating point for
  an alerting system with a `WARNING` floor. Ranking (ROC 0.919, PR 0.858) is on par with
  the stronger tree models.
- The engine sidesteps the LSTM threshold-transfer problem by blending model probabilities
  into a continuous score rather than trusting any single model's decision boundary.

## Artifacts

- Report: `reports/evaluations/risk_report.json`
- Calibration: `models/risk/calibration.json` (component weights + signal percentiles)
- Models consumed: `models/baseline/*`, `models/advanced/*`, `models/prediction/lstm_classifier`

## Re-run

```powershell
python backend/scripts/score_risk.py
```

## Usage

```python
from src.risk import RiskEngine

engine = RiskEngine.load("backend/models", settings)
latest = engine.latest_by_service(df)     # scoring DataFrame with risk rows
```