# Autoencoder Anomaly Detection

**Day 8** · Goal: replace the univariate/forest baselines with a learned reconstruction
model — train an autoencoder on **normal data only** and use reconstruction error as the
anomaly score.

## Deliverables

- `src/anomaly/autoencoder.py`
  - `DenseAutoencoderDetector` — fully-connected bottleneck autoencoder on the 26 features.
  - `LSTMAutoencoderDetector` — sequence autoencoder over per-service 10-minute windows.
  - shared `_AutoencoderBase` (normal-only filtering, scaling, per-service models,
    threshold, joblib/`.keras` save + load).
- `scripts/score_anomaly.py --include-lstm` — fits on train, scores val/test, writes
  `data/features/anomaly_scores.parquet` (33,960 rows) and
  `reports/evaluations/anomaly_baseline_report.json`.
- `tests/test_autoencoder.py` — 4 tests (outlier ranking, normal-only mask, save/load
  round-trip, LSTM finite scores).
- Output: **autoencoder anomaly scores** for every validation and test row.

## Design

- **Trained only on normal data**: rows where `in_failure == 0` and
  `failure_in_next_10min == 0` (`normal_only: true`). The model never sees a failure or its
  10-minute ramp, so the pre-failure rise is genuinely out-of-distribution.
- **Scaling**: `StandardScaler` statistics (mean/std) fitted on normal rows only; stored
  with the model so scoring is consistent.
- **Per service**: one autoencoder per service (global fallback for unseen services),
  matching the other Module 3 detectors.
- **Dense architecture**: `26 → 32 → 8 → 32 → 26`, ReLU encoder/decoder, linear output,
  Adam `1e-3`, MSE, batch 256, up to 50 epochs with early stopping (patience 5) on a 10%
  internal validation split. Score = per-row MSE.
- **LSTM architecture**: `seq_len=20` (10 min at 30 s) → `LSTM(32)` → `RepeatVector(20)` →
  `LSTM(32, return_sequences)` → `TimeDistributed(Dense(26))`. Training windows are kept
  only when **every** tick in the window is normal (no sequence spans a failure). Window MSE
  is attributed to the window's last tick; warm-up rows reuse the first window's error.
- **Threshold**: `1 - contamination` (0.05) quantile of normal training reconstruction
  error, calibrated in `fit()` before the generic scoring pass.

## Results — validation split (`failure_in_next_10min`, positive rate 0.097)

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| zscore | 0.8678 | 0.5279 | 0.5761 | 0.5889 | 0.5824 |
| iqr | 0.8952 | 0.7575 | 0.8908 | 0.3926 | 0.5450 |
| rolling_iqr | 0.6782 | 0.4876 | 0.7883 | 0.4000 | 0.5307 |
| isolation_forest | 0.8996 | 0.7462 | 0.5850 | 0.7519 | 0.6580 |
| **dense autoencoder** | **0.9437** | **0.8789** | **0.9412** | 0.7704 | **0.8473** |
| lstm autoencoder | 0.8582 | 0.6799 | 0.8077 | 0.3889 | 0.5250 |

The dense autoencoder is the strongest detector on every metric except recall, improving
validation ROC-AUC from 0.90 → **0.94** and F1 from 0.66 → **0.85** over the best baseline.
Learning the joint normal manifold captures the multi-feature failure signature
(memory growth + latency + error rate + pool usage) that single-feature z/IQR scores miss.

The **LSTM autoencoder underperforms here** (ROC-AUC 0.86). The inputs are already
5/10/15-minute rolling aggregates, so a 10-minute sequence adds little new information while
the small LSTM has far fewer effective samples and trains less stably. It is kept as a
time-series reference; the dense model is the one to carry into Modules 4–5.

## Usage

```python
from src.anomaly import DenseAutoencoderDetector
from src.features import FeatureStore

store = FeatureStore("data/features")
meta = store.read_metadata()
splits = store.read_splits("data/splits")

model = DenseAutoencoderDetector(hidden_dims=(32, 8), epochs=50).fit(splits["train"], meta["feature_columns"])
scores = model.score(splits["val"])          # reconstruction error, higher = worse
model.save("models/anomaly/dense_autoencoder")
restored = DenseAutoencoderDetector.load("models/anomaly/dense_autoencoder")
```

```powershell
& "C:\Users\SHIVANESH\miniconda3\envs\sentinelai\python.exe" "scripts\score_anomaly.py" --include-lstm
& "C:\Users\SHIVANESH\miniconda3\envs\sentinelai\python.exe" -m unittest tests.test_autoencoder -v
```

## Notes / decisions

- TensorFlow 2.21 on CPU; `TF_CPP_MIN_LOG_LEVEL=2` and `keras.utils.set_random_seed(42)`
  keep runs deterministic and quiet.
- Serialisation is a directory: `model_<service>.keras` per service plus `meta.pkl`
  (scaler, threshold, config). Verified by a save/load round-trip test.
- AE scores share the `higher = more anomalous` contract and the same severity-level
  mapping as the Day 7 baselines, so Module 5 can consume them uniformly.
- Training is quick (< ~1 min dense, a few minutes LSTM for 5 services) and the fitted
  models are not committed; they are reproducible from seed 42.
