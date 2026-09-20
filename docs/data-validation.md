# Data Pipeline Validation

**Day 4** · Goal: produce and validate a multi-day, multi-service telemetry dataset that
cleanly separates normal operation, gradual degradation, sudden failure, and recovery, with
ground-truth incident labels suitable for supervised anomaly detection and
failure-in-X-minutes forecasting.

## Approach

1. **Multi-day coverage** — the 12h scenario schedule from `config/settings.yaml`
   (`simulation.scenario_schedule`) is tiled by `simulation.schedule_repeats` and
   `simulation.schedule_period_min`. With `repeats: 2` and `period: 720`, the default 24 h
   run contains two copies of each scenario type at different times of day (diurnal background
   load differs between tiles).
2. **Shared config** — `scripts/generate_data.py` and `scripts/collect_telemetry.py` now
   source settings, services, scenarios, and the default duration from
   `src/telemetry/config.py` (single source of truth; inline loaders removed).
3. **Uniform reader** — `src/telemetry/reader.py` discovers raw data (CSV / JSONL /
   Parquet, flat files or `dt=*` hourly partitions) and returns columns matching the
   telemetry/log/ground-truth headers. Files are matched by name prefix
   (`metrics.*`, `logs.*`, `ground_truth.*`) so partition layouts stay interchangeable.
4. **Checks** — `src/telemetry/validation.py` runs 21 checks (see below); the CLI in
   `scripts/validate_data.py` writes a machine-readable report to
   `reports/evaluations/validation_report_<timestamp>.json` and prints a summary table.

## Exit criteria (from the plan)

- Multi-day data covering normal operation, gradual degradation, sudden failure, and recovery. **Met.**
- Sanity checks on distributions, missing values, alignment, and bounds. **Met.**
- Deterministic re-runs from the same seed. **Met** (`simulation.seed: 42`).
- Dataset is split ready (raw stores sidecar metadata; partitioning by hourly `dt`). **Met.**

## Dataset (canonical, regenerated)

| Resource        | Rows      | Notes                                        |
|-----------------|-----------|----------------------------------------------|
| Metrics         | 14,400    | 5 services x 2,880 ticks @ 30 s = 24 h       |
| Logs            | 4,998     | INFO / WARNING / ERROR, correlated           |
| Ground truth    | 10        | 5 scenario types x 2 occurrences             |

Incidents: `memory_leak` (2x120m), `connection_pool_exhaustion` (2x60m), `latency_spike`
(2x20m), `network_partition` (2x25m), `disk_fill` (2x120m). `auth-service` stays healthy by
design (no scenario assigned), giving the dataset a known open-set service.

## Validation results (overall **PASS**)

### Structure
- 15-column metrics header matches `TELEMETRY_HEADER`; 4-column logs, 5-column
  ground-truth headers exact.
- 2,880 ticks per service, identical timestamp sets across all 5 services, all 2,879 gaps
  exactly 30 s.
- No NaN / Inf in the 12 numeric metrics; every metric within `METRIC_BOUNDS`.

### Failure realism
- `healthy` share 0.917; degraded ticks cluster inside incident windows (0.855 of ticks in
  windows unhealthy, well above the 0.6 threshold).
- Services return to healthy immediately after each incident end (recovery verified).
- All 10 incidents have WARNING/ERROR logs inside their window (log/metric/trace aligned).

### Ground truth
- Every planned scenario type appears exactly twice; durations match the schedule (PASS,
  +/-2 samples); no overlapping incidents per service.

### Informational
- Metric distributions (mean/std/min/p50/max) archived in the JSON for feature-engineering
  sanity checks; latency spikes to ~1.1 s and request-rate maxima ~391 rps confirm
  incident signatures are present in the numeric profile.

## Checks list (21)

`metrics_schema`, `logs_schema`, `truth_schema`, `per_service_ticks`,
`timestamp_alignment`, `uniform_spacing`, `missing_values`, `metric_bounds`,
`healthy_distribution`, `per_service_failures` (INFO), `log_levels`, `log_time_range`,
`incident_types`, `incident_duration`, `incident_overlap`, `incident_schedule_match`,
`unhealthy_alignment`, `recovery_after_incident`, `log_correlation`, `expected_ticks` (INFO),
`metric_range` (INFO), `incident_rows` (INFO).

Overall = FAIL if any FAIL, else WARN if any WARN, else PASS.

## Usage

```powershell
& "C:\Users\SHIVANESH\miniconda3\envs\sentinelai\python.exe" "scripts\generate_data.py"     # 24 h default
& "C:\Users\SHIVANESH\miniconda3\envs\sentinelai\python.exe" "scripts\validate_data.py"     # report + table
```

Optional `--duration-min`, `--seed`, `--interval-sec`, `--format`, `--out` flags on the
generator; `--root`, `--settings`, `--interval-sec`, `--report-dir` on the validator.

## Notes / decisions

- Pandas 3.0 parses `True/False` as `bool` dtype, so the `healthy` column is excluded from
  the numeric metric blocks (distributions, bounds) to avoid `describe()` dropping it.
- The reader matches files by name prefix, so `data/raw/metrics` may legally contain
  `metrics.csv` and `ground_truth.csv` (batch generator writes both there); partition
  collectors write `data/raw/truth/dt=*` instead, and `load_raw` prefers `raw/truth`.