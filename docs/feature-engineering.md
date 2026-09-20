# Feature Engineering Pipeline

**Day 5 (Module 2)** · Goal: turn validated raw telemetry into a reusable, causal
feature table for anomaly detection (Module 3) and failure forecasting (Module 4).

## Deliverable

- `src/features/engineering.py` — `FeatureEngineer` class + `build_features()` /
  `build_features_from_raw()` functions (reusable module, not notebook code).
- `src/features/__init__.py` — public exports.
- `scripts/build_features.py` — CLI that reads `data/raw` and writes
  `data/features/features.{parquet,csv}`.
- `tests/test_features.py` — 14 stdlib `unittest` tests (correctness, causal safety,
  service isolation, config).
- Config block `features:` in `config/settings.yaml`.

## Feature set (26 features)

All features are computed **per service** and are **causal**: a feature at time `t` uses
only ticks `<= t`, so it is safe for forecasting without look-ahead leakage.

### Rolling averages — `{Metric}_{N}min_avg`
For `N in {5, 10, 15}` over: `cpu_util_pct` (CPU), `memory_util_pct` (Memory),
`latency_ms` (Latency), `disk_util_pct` (Disk), `db_connection_usage_pct` (DB_usage),
`network_in_mbps` (Network_in) → 18 features, including the required
`CPU_5min_avg`, `Memory_10min_avg`, `Latency_5min_avg`.

### Growth rates — `{Metric}_growth_rate`
Relative change over a 10-minute lookback: `(x_t - x_{t-W}) / |x_{t-W}|` for CPU, Memory,
Disk, DB_usage → `CPU_growth_rate`, `Memory_growth_rate`, `Disk_growth_rate`,
`DB_usage_growth_rate`. Warm-up rows (no prior value) are `0.0`; division is floored at
`eps` and `±inf` is neutralised.

### Cross-metric / derived
| Feature | Definition | Window |
|---|---|---|
| `Error_rate` | rolling mean of `http_4xx_rate + http_5xx_rate` | 5 min |
| `Request_rate` | rolling mean of `request_rate_rps` | 5 min |
| `CPU_memory_ratio` | `cpu_util_pct / memory_util_pct` | instant |
| `latency_per_request` | `latency_ms / max(request_rate_rps, 1.0)` | instant |

`latency_per_request` floors the denominator at `request_floor_rps` to avoid blow-ups at
low overnight traffic.

## Configuration (`config/settings.yaml`)

```yaml
features:
  windows_min: [5, 10, 15]
  growth_window_min: 10
  error_window_min: 5
  request_window_min: 5
  min_periods: 1          # partial windows allowed during warm-up (no NaNs)
  request_floor_rps: 1.0
```

`feature_config_from_settings()` also falls back to
`telemetry.collection_interval_sec` and `telemetry.feature_windows_min`. Window sizes in
minutes are converted to ticks at 30 s → 10 / 20 / 30 ticks.

## Usage

```python
from src.features import build_features_from_raw, FeatureConfig

features = build_features_from_raw("data")          # 14400 x 26, 0 NaN
features = build_features_from_raw("data", FeatureConfig(windows_min=[5, 10]))
```

```powershell
& "C:\Users\SHIVANESH\miniconda3\envs\sentinelai\python.exe" "scripts\build_features.py"
& "C:\Users\SHIVANESH\miniconda3\envs\sentinelai\python.exe" -m unittest tests.test_features -v
```

## Verification

- Unit tests: **14/14 OK** — required names, rolling-mean values, growth-rate math,
  ratio/error features, per-service isolation, and an explicit **no-look-ahead test**
  (perturbing the final tick leaves all features at `t <= 59 - 30` unchanged).
- Full dataset: `14400 rows x 26 features`, `NaN = 0`, 2,880 rows per service.
- Incident discrimination on the canonical 24 h dataset (payment-api):

| Scenario | Feature | In-window | Baseline |
|---|---|---|---|
| memory_leak | `Memory_growth_rate` | 0.027 | 0.000 |
| memory_leak | `Memory_10min_avg` | 61.8 | 51.9 |
| connection_pool_exhaustion | `DB_usage_10min_avg` | 60.6 | 43.2 |
| connection_pool_exhaustion | `Error_rate` | 0.031 | 0.004 |

Cross-metric and growth features add clear separation beyond raw instantaneous metrics,
confirming the pipeline feeds useful signal downstream.

## Notes / decisions

- Rolling windows are past-inclusive (`pandas.rolling`), not centered — no future data.
- `min_periods: 1` keeps the matrix NaN-free; the first few ticks use partial windows and
  are covered by a dedicated unit test.
- Service isolation is guaranteed by grouping on `service` before every `rolling` /
  `shift`, so one service's incident cannot contaminate another's features.
- Output includes `timestamp`, `service`, and passthrough `healthy`; `healthy` is a label
  source, not a model feature, and should be excluded by the trainer.
