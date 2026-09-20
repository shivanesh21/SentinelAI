# Baseline Models (Module 4)

## Overview
Trained supervised classifiers on the labeled feature set to predict `failure_in_next_10min` (binary classification: failure within the next 10 minutes). Two baseline models were evaluated: **Logistic Regression** and **Random Forest**.

## Data
- **Features**: 26 engineered features per service (rolling averages, growth rates, ratios, error/request rates)
- **Target**: `failure_in_next_10min` (1 if any incident starts in the next 10 min / 20 ticks)
- **Splits**: Chronological time-based split with 20-tick purge (embargo)
  - Train: 8,540 rows (positive rate 11.15%)
  - Val: 2,780 rows (positive rate 9.71%)
  - Test: 2,880 rows (positive rate 11.39%)

## Model Configurations

### Logistic Regression
- Pipeline: `StandardScaler` → `LogisticRegression`
- Penalty: L2 (C=1.0)
- Class weight: `balanced`
- Solver: `lbfgs`, max_iter=1000

### Random Forest
- Pipeline: `StandardScaler` → `RandomForestClassifier`
- n_estimators: 200
- max_depth: 12
- min_samples_split: 10
- min_samples_leaf: 4
- Class weight: `balanced_subsample`
- n_jobs: -1

## Threshold Selection
Threshold chosen on validation set to maximize **F1-score** (using precision-recall curve).

## Results Summary

| Model | Split | ROC-AUC | PR-AUC | Precision | Recall | F1 | Threshold |
|-------|-------|---------|--------|-----------|--------|-----|-----------|
| Logistic Regression | Val | **0.9593** | **0.8982** | 0.9698 | 0.8333 | 0.8964 | 0.7042 |
| Logistic Regression | Test | 0.9155 | 0.8971 | 0.7639 | 0.8780 | 0.8170 | 0.7042 |
| Random Forest | Val | 0.9158 | 0.8541 | **0.9954** | 0.7963 | 0.8848 | 0.6555 |
| Random Forest | Test | 0.8990 | 0.8387 | 0.9583 | 0.7713 | 0.8547 | 0.6555 |

## Key Observations
1. **Logistic Regression** achieves higher ROC-AUC (0.959 vs 0.916 on val) and better recall, making it more sensitive to impending failures.
2. **Random Forest** achieves higher precision (0.995 vs 0.970 on val) with fewer false alarms, but lower recall.
3. On test, Random Forest maintains better precision (0.958 vs 0.764) but Logistic Regression has better recall (0.878 vs 0.771).
4. Both models generalize well from val to test with minimal degradation in ranking metrics (ROC-AUC/PR-AUC).

## Feature Importance (Top 5)

### Logistic Regression (by absolute coefficient magnitude)
1. `Memory_5min_avg` (+5.60)
2. `Latency_15min_avg` (-4.26)
3. `Request_rate` (-4.20)
4. `Latency_5min_avg` (+3.84)
5. `Error_rate` (+2.77)

### Random Forest (by Gini importance)
1. `Latency_5min_avg` (0.150)
2. `Latency_10min_avg` (0.105)
3. `Latency_15min_avg` (0.098)
4. `Error_rate` (0.076)
5. `Memory_5min_avg` (0.064)

**Interpretation**: Latency features dominate Random Forest importance, while Logistic Regression weighs memory utilization and latency differently. Both models identify latency and memory as primary failure precursors.

## Artifacts
- Models persisted to `models/baseline/{logistic_regression,random_forest}/`
  - `model.joblib` — fitted sklearn Pipeline
  - `meta.json` — threshold, feature columns, metadata
- Evaluation report: `reports/evaluations/baseline_report.json`

## Usage
```python
from src.models.baseline import load_model

pipe, threshold, feature_columns, meta = load_model("models/baseline/logistic_regression")
probs = pipe.predict_proba(df[feature_columns])[:, 1]
preds = (probs >= threshold).astype(int)
```

## Next Steps (Module 5 — Risk Scoring)
- Combine baseline failure probability with anomaly scores (Day 9) and other signals
- Calibrate multi-signal risk score with severity thresholds
- Evaluate lead time and actionability for auto-remediation triggers