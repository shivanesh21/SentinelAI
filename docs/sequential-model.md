# Time-Series Deep Learning Model (Module 4b — LSTM)

## Overview

Day 12 deliverable: an **LSTM binary classifier** over sequential windows of the labeled
feature table, predicting `failure_in_next_10min`, compared against the Day 10/11
tree-based models (Logistic Regression, Random Forest, XGBoost, LightGBM) on the
**same chronological test split** and the **same 26 features**.

## Window construction (leak-free)

- Per service, time-ordered rows are sliced into sliding windows of `seq_len` consecutive
  ticks ending at the prediction tick.
- The target is the `failure_in_next_10min` label of the window's **last** row.
- Windows are built independently per split, so a window never straddles a split boundary
  (no look-ahead leakage), and never mixes services.
- The `seq_len - 1` warm-up rows of each service in each split are dropped.
- Features are standardised per column with a scaler fitted **on training windows only**.

Window counts:

| Split | Windows | Positives | Positive rate |
|-------|---------|-----------|---------------|
| Train | 8,445 | 952 | 11.27% |
| Val   | 2,685 | 251 | 9.35% |
| Test  | 2,785 | 328 | 11.58% |

## Architecture

```
Input (seq_len, 26)  →  LSTM(32, dropout=0.2)  →  Dense(16, relu)  →  Dropout(0.2)  →  Dense(1, sigmoid)
```

- Optimizer: Adam (lr 1e-3), loss: binary crossentropy, metric: AUC.
- Class weights `{0: 1.0, 1: 7.92}` compensate for the ~11% positive rate.
- Early stopping on `val_auc` (patience 5, restore best weights).

## Hyperparameter tuning

Six candidates scanned on **validation F1** (early-stopped; sequences rebuilt per config):

| Config (seq_len, units, dropout, lr) | Val F1 | Val ROC |
|---------------------------------------|--------|---------|
| **10, 32, 0.2, 1e-3 (selected)**      | **0.8392** | **0.9840** |
| 20, 48, 0.3, 1e-3 (batch 64)          | 0.8248 | 0.9809 |
| 20, 32, 0.2, 1e-3                     | 0.8197 | 0.9825 |
| 30, 32, 0.2, 1e-3                     | 0.8162 | 0.9820 |
| 20, 32, 0.1, 5e-4                     | 0.8172 | 0.9817 |
| 20, 64, 0.2, 1e-3                     | 0.8151 | 0.9745 |

Best config by val F1: `seq_len=10, lstm_units=32, dropout=0.2, lr=1e-3, batch=128`.

## Results (same test split, val-optimized threshold)

| Model | F1 | Precision | Recall | ROC-AUC | PR-AUC |
|-------|-----|-----------|--------|---------|--------|
| **Random Forest** | **0.8547** | **0.9583** | 0.7713 | 0.8990 | 0.8387 |
| Logistic Regression | 0.8170 | 0.7639 | **0.8780** | 0.9155 | **0.8971** |
| XGBoost | 0.7487 | 0.8521 | 0.6677 | 0.9181 | 0.8229 |
| LightGBM | 0.7166 | 0.7692 | 0.6707 | **0.9289** | 0.8217 |
| **LSTM (tuned)** | 0.4731 | 0.3329 | 0.8171 | 0.9085 | 0.8334 |

Validation: LSTM F1 **0.8392**, ROC-AUC **0.9840**, PR-AUC **0.9126** (best of all models on val).

## Analysis

1. **Ranking quality is competitive.** Test ROC-AUC (0.908) and PR-AUC (0.833) sit mid-pack
   among the tree models; the LSTM clearly ranks failure windows ahead of healthy ones.

2. **The val-optimized decision threshold does not transfer to the test period.** The LSTM
   assigns 28–30% of test windows a score above the val threshold vs an 11.6% actual positive
   rate; trees transfer cleanly (RF: 9.2% predicted vs 11.6% actual). Symptom of score
   distribution shift between the two simulated schedule tiles (baseline traffic / time of day).
   Re-scoring validation with a "rate-matched" threshold (predicted positives == observed
   positives) yields test F1 ≈ 0.78, showing most of the F1 gap is threshold calibration loss
   rather than pure scorer weakness.

3. **Takeaway for the platform:** the LSTM is a strong *continuous risk signal* but a weak
   *calibrated alarm* under drift. Trees remain the primary classifier for thresholded alerts;
   the LSTM probability should be consumed by the Module 5 risk engine (continuous score
   fusion), where calibration/domain-adaptation can be handled at the aggregation layer.

## Artifacts

- Model: `models/prediction/lstm_classifier/` (`model.keras` + `meta.pkl` with config,
  scaler, class weight, training history)
- Report: `reports/evaluations/sequential_report.json` (config, tuning table, val/test
  metrics, cross-model comparison)

## Usage

```python
from src.models.sequential import SequentialLSTMClassifier, build_sequences

model = SequentialLSTMClassifier.load("models/prediction/lstm_classifier")
X, y, meta = build_sequences(df, feature_columns, seq_len=model.config.seq_len)
probs = model.predict_proba(X)   # continuous score for the risk engine
```

## Re-run

```powershell
python backend/scripts/train_sequential.py
```