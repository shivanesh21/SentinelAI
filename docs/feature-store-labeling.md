# Feature Store & Labeling

**Day 6** · Goal: persist the engineered features as a per-service, per-timestamp feature
table, attach supervised labels from the injected failure windows, and split the data
chronologically into train/validation/test.

## Deliverables

- `src/features/labels.py` — `build_labels()` / `build_feature_labels()` + `LabelConfig`.
- `src/features/splits.py` — `time_based_split()` + `SplitConfig` / `SplitReport`.
- `src/features/store.py` — parquet-backed `FeatureStore` with JSON metadata.
- `scripts/build_dataset.py` — one command builds the labeled feature table and splits.
- `tests/test_labels.py`, `tests/test_splits.py` — 14 tests.
- Outputs: `data/features/feature_table.parquet` (+ `.meta.json`),
  `data/splits/{train,val,test}.parquet` (+ `split_report.json`),
  `reports/evaluations/dataset_report.json`.

## Feature table

14400 rows × 26 features + 3 labels. Schema (`timestamp`, `service`, per-service per-tick):

| Column | Meaning |
|---|---|
| `failure_in_next_10min` | target: service in failure at `t` or at any tick within the next 10 min |
| `in_failure` | service is currently inside an incident window `[start, end)` |
| `time_to_failure_min` | minutes to the next incident onset, `0` while failing, `NaN` if none remains |
| 26 feature columns | outputs of the Day 5 `FeatureEngineer` |

Label definition uses the generator's ground truth directly: `start_ts` is the first
failing tick, `end_ts` the first healthy tick, so the active window is `[start_ts, end_ts)`.

`failure_in_next_10min[t]` is the forward max of the failing indicator over the next
`horizon` ticks and is a **target**, so it must never be used as an input feature.
`in_failure` and `time_to_failure_min` are auxiliary / analysis columns, not model inputs.
The `FeatureTableMeta` records the exact `feature_columns` and `label_columns` so the
trainer cannot accidentally include a target.

Positive rate: 1,570 / 14,400 = **0.109**.

## Time-based splits

Random shuffling is invalid for time series. `time_based_split()` cuts on the unique
timestamp grid (identical boundaries for every service) with fractions 0.6 / 0.2 / 0.2,
then applies a **purge/embargo** of `purge_min` (default = label horizon = 10 min → 20
ticks) at each boundary so a training label whose 10-minute future window reaches into the
validation block cannot leak.

| Split | Rows | Positive rate | Window |
|---|---|---|---|
| train | 8,540 | 0.1115 | up to 2026-09-18T21:19 |
| val | 2,780 | 0.0971 | 2026-09-18T21:29 → 2026-09-19T02:07 |
| test | 2,880 | 0.1139 | from 2026-09-19T02:17 |

Verified: strictly chronological (no overlap, `max(train) < min(val) < max(val) < min(test)`),
all 5 services present in every split, and both classes present in every split. Because the
scenario schedule is tiled twice, val and test each still contain real incidents.

## Usage

```python
from src.features import FeatureStore, build_feature_labels, time_based_split

store = FeatureStore("data/features")
table = store.read_table()                      # 14400 x 29 + metadata
meta = store.read_metadata()                    # feature_columns / label_columns
splits = store.read_splits("data/splits")       # train / val / test
```

```powershell
& "C:\Users\SHIVANESH\miniconda3\envs\sentinelai\python.exe" "scripts\build_dataset.py"
& "C:\Users\SHIVANESH\miniconda3\envs\sentinelai\python.exe" -m unittest tests.test_labels tests.test_splits -v
```

## Notes / decisions

- Pandas 3 stores tz-aware timestamps as `datetime64[us, UTC]`; `.astype("int64")` yields
  **microseconds**. Label math converts via `.to_numpy(dtype="datetime64[ns]")` to force
  nanoseconds, otherwise `time_to_failure_min` was off by 1000x.
- Purge gap is a true gap: the rows between `train_end` and `val_start` are dropped, not
  reassigned, so no sample sits within a label horizon of a boundary.
- The splitter validates enough distinct timestamps and reports `chronological` /
  `overlap_rows` flags for downstream CI to assert on.
