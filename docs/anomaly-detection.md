# Anomaly Detection Baseline (Module 3)

**Day 7** · Goal: establish unsupervised anomaly-detection baselines on the engineered
feature table so later models (predictive classifier, autoencoder, risk scorer) have a
reference to beat.

## Deliverables

- `src/anomaly/base.py` — `BaseDetector` interface (`fit` / `score` / `predict` /
  `fit_threshold` / `score_frame` / `save` / `load`) + `assign_levels()`.
- `src/anomaly/statistical.py` — `ZScoreDetector`, `IQRTukeyDetector`, `RollingIQRDetector`.
- `src/anomaly/isolation_forest.py` — `IsolationForestDetector` (sklearn, per service).
- `scripts/score_anomaly.py` — fits on train, scores validation/test, writes scores + report.
- `tests/test_anomaly.py` — 5 tests (outlier detection, rolling scores finite, IForest
  save/load round-trip, monotonic severity levels).
- Outputs: `data/features/anomaly_scores.parquet`,
  `reports/evaluations/anomaly_baseline_report.json`.

## Design

- All detectors return a continuous score where **higher = more anomalous**.
- One detector is fitted **per service** (`group_col="service"`) so each service is
  compared against its own baseline; a global fallback covers unseen services.
- The decision threshold is the `1 - contamination` quantile of **training** scores
  (`contamination = 0.05`), so no validation/test information influences the cut.
- Feature inputs are the 26 engineered features from the feature-table metadata; labels
  and `time_to_failure_min` are never used as inputs.

| Detector | Scoring |
|---|---|
| `zscore` | per-feature train mean/std, score = `max_feature |z|` |
| `iqr` | per-feature train Q1/Q3, score = max normalized excursion beyond Tukey fences |
| `rolling_iqr` | per-service rolling (60 min) median/Q1/Q3 fences, excursion of the current value |
| `isolation_forest` | 200 trees, score = `-score_samples` |

## Results on the validation split (primary deliverable)

Scored against the held-out `failure_in_next_10min` label (positive rate 0.097), threshold
fixed from training scores at 5% contamination:

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| zscore | 0.8678 | 0.5279 | 0.5761 | 0.5889 | 0.5824 |
| iqr | 0.8952 | 0.7575 | 0.8908 | 0.3926 | 0.5450 |
| rolling_iqr | 0.6782 | 0.4876 | 0.7883 | 0.4000 | 0.5307 |
| **isolation_forest** | **0.8996** | 0.7462 | 0.5850 | **0.7519** | **0.6580** |

Interpretation:

- **Isolation Forest** is the strongest overall baseline (best ROC-AUC and F1) and is the
  one to carry forward; it captures non-linear feature interactions the univariate
  baselines miss.
- **IQR** is very precise but conservative — useful when false alarms are expensive.
- **Rolling IQR** underperforms because the inputs are already rolling aggregates, so
  re-smoothing them over a 60-minute window blurs the 10-minute failure ramp. Its value is
  as a transparent, distribution-free sanity check.

Anomaly scores and a four-level severity label (`NORMAL` / `WARNING` / `ANOMALOUS` /
`CRITICAL`, calibrated on the training score distribution) are persisted for every
validation and test row, per model.

## Usage

```python
from src.anomaly import ZScoreDetector, IsolationForestDetector
from src.features import FeatureStore

store = FeatureStore("data/features")
meta = store.read_metadata()
splits = store.read_splits("data/splits")

model = IsolationForestDetector(contamination=0.05).fit(splits["train"], meta["feature_columns"])
model.fit_threshold(model.score(splits["train"]))
scores = model.score(splits["val"])
```

```powershell
& "C:\Users\SHIVANESH\miniconda3\envs\sentinelai\python.exe" "scripts\score_anomaly.py"
& "C:\Users\SHIVANESH\miniconda3\envs\sentinelai\python.exe" -m unittest tests.test_anomaly -v
```

## Notes / decisions

- Scores are model-agnostic (higher = worse), so Module 5 can consume them directly or
  re-calibrate per detector.
- Severity thresholds use train-score quantiles (50% / `1-contamination` / `1-contamination/2`),
  giving roughly 50 / 5 / 2.5 % of normal traffic in the upper buckets.
- `IsolationForestDetector` is serialisable with joblib and verified by a round-trip test;
  fitted baseline models are not yet committed (retrained deterministically from seed 42).
