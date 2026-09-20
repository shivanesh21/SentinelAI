# SentinelAI — Telemetry Schema & Failure-Injection Design

Deliverable for Day 2 (Module 1). Documents the synthetic telemetry format and how
controlled failures are injected to produce labeled ground truth.

## 1. Telemetry Schema

Metrics are emitted per service per sampling tick (default 30 s). A metric row:

| Field | Type | Units | Range | Meaning |
|---|---|---|---|---|
| `timestamp` | ISO-8601 UTC | — | — | Sampling time |
| `service` | str | — | — | Affected service name |
| `cpu_util_pct` | float | % | [0, 100] | CPU utilization |
| `memory_util_pct` | float | % | [0, 100] | Memory utilization |
| `disk_util_pct` | float | % | [0, 100] | Disk utilization |
| `network_in_mbps` | float | Mbps | ≥ 0 | Inbound throughput |
| `network_out_mbps` | float | Mbps | ≥ 0 | Outbound throughput |
| `request_rate_rps` | float | req/s | ≥ 0 | Incoming request rate |
| `latency_ms` | float | ms | ≥ 0 | P95 proxy: response latency |
| `http_4xx_rate` | float | fraction | [0, 1] | Client-error share of requests |
| `http_5xx_rate` | float | fraction | [0, 1] | Server-error share of requests |
| `db_connections` | float | count | ≥ 0 | Active DB connections |
| `db_connection_usage_pct` | float | % | [0, 100] | DB connection usage of pool |
| `container_restarts` | float | count | ≥ 0 | Cumulative crash-restarts |
| `healthy` | bool | — | — | Service health flag |

**Log format** (`data/raw/logs/`):

| Field | Type | Notes |
|---|---|---|
| `timestamp` | ISO-8601 UTC | |
| `service` | str | |
| `level` | INFO / WARNING / ERROR | enum |
| `message` | str | e.g. `database connection timeout`, `connection pool exhausted` |

**Ground truth** (`data/raw/metrics/ground_truth.*`): injected incident windows.

| Field | Type | Notes |
|---|---|---|
| `incident_id` | int | Sequential |
| `service` | str | |
| `scenario_type` | str | `memory_leak` … `disk_fill` |
| `start_ts` / `end_ts` | ISO-8601 UTC | Failure window (label source) |

### Output formats
- **CSV**: one header + one row per record (`data/raw/metrics/metrics.csv`, `…/logs.csv`, `…/ground_truth.csv`).
- **JSONL**: same records, one JSON object per line.
Generated via `scripts/generate_data.py --format csv|jsonl --duration-min N`.

## 2. Baseline behavior (normal operation)

Each service has a profile (`config/services.yaml`) and emits a base signal with:
- **Diurnal load**: request rate/latency scale with a 24 h sinusoid peaking mid-day.
- **Service-specific baselines** (CPU, memory, latency, request rate, DB pool size).
- **Stochastic noise** (seeded, reproducible via `--seed`).

Normal logs are INFO/WARNING heartbeats at low frequency.

## 3. Failure-injection design

Each scenario is a function of elapsed time over its window and **overrides** specific
metrics, producing realistic degradation curves rather than instant threshold jumps.

| `scenario_type` | Metric signature (visible before/at failure) | Correlated logs |
|---|---|---|
| `memory_leak` | memory ramps +up to 30 pts → latency rises; risky beyond ~90% | `WARNING memory usage increasing` → `ERROR memory pressure critical: OOM risk detected` |
| `connection_pool_exhaustion` | db usage 30%→100%; connections ∝ usage; latency ×(1+2.5·ramp); 5xx rises near cap | `ERROR database connection timeout`, `ERROR connection pool exhausted`, `WARNING retry limit exceeded` |
| `latency_spike` | latency ×(1+9·intensity) fast; 5xx rises; request rate drops under backpressure; 1-2 crash-restarts at end | `WARNING response latency increasing` → `ERROR service unavailable`, `WARNING retry attempt` |
| `network_partition` | wave-driven packet drops: request rate collapses, latency ×5, 4xx/5xx spike; partitions oscillate | `WARNING retry attempt`, `ERROR connection timeout`, `WARNING network flapping detected` |
| `disk_fill` | disk ramps to ~99%; beyond 90% writes fail → 5xx and latency rise | `WARNING disk utilization high` → `ERROR disk write failed: no space left` |

**Lifecycle phases** per scenario: `degrading → failed → recovering`, computed from the
elapsed ratio. A service is `healthy=false` during `degrading`/`failed` phases.

### Ground-truth labeling
- Start = first tick inside a scenario window; end = first tick after it.
- This directly supports the Day 5–6 label: `failure_in_next_10min = 1` for any row
  within 10 min *before* a failure window.

### Config surface
- Scenario schedule: `config/settings.yaml → simulation.scenario_schedule`
  (`service`, `type`, `start_min`, `duration_min`, optional `intensity`).
- Per-service profiles: `config/services.yaml → services[i].profile`.

## 4. Reproducibility
- Fixed `--seed` (default 42) + fixed `--start-ts` → identical output byte-for-byte.
- The default 12 h run produces: 7 200 metric rows, 5 labeled incidents, ~2 600 log lines.