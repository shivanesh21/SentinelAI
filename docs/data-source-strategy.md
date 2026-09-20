# SentinelAI — Data Source Strategy

## Decision: Simulated (Synthetic) Telemetry

**Rationale.** A 25-day timeline and an ML component require **labeled ground truth** for
failure prediction and evaluation. Real clusters rarely produce failures on demand, are
difficult to label, and introduce environment setup overhead.

We therefore run a **telemetry simulator** (`src/telemetry/simulator`) that emulates a set of
microservices for which we control ground truth exactly.

## What the simulator produces

Per service (see `config/services.yaml`), at a 30 s interval:
- CPU, memory, disk, network IO
- Request rate, response latency, HTTP 4xx/5xx rates
- DB connection count/usage, container restarts
- Log lines (INFO/WARNING/ERROR) correlated with the injected failure condition

## Failure-injection scenarios (labeled ground truth)

| Scenario | Mechanism |
|---|---|
| `memory_leak` | Memory grows monotonically; latency degrades near saturation |
| `connection_pool_exhaustion` | DB connections climb to 100% of pool; timeout errors rise |
| `latency_spike` | Sudden latency jump; 5xx increase; recovery after restart |
| `network_partition` | Intermittent request drops; timeouts; worker retries |
| `disk_fill` | Disk utilization creeps up; writes begin to fail |

Each injection session is recorded with timestamps for failure start/end, so labels
`failure_in_next_10min` can be generated exactly (Day 5–6).

## Storage

- Raw: partitioned CSV/Parquet under `data/raw/metrics` and `data/raw/logs`.
- Engineered features: `data/features`. Labels: `data/labels`. Splits: `data/splits`.
- No external database — matches a single-host student-project scope while leaving the schema
  compatible with Prometheus/TimescaleDB later.

## Why not real infrastructure (Y1)
- No access to a real Kubernetes cluster.
- Hard to force deterministic failures for evaluation.
- Simulation keeps the demo reproducible: a failure can be injected on demand for the live
  demo (inject → detect → RCA → remediate → recover).

## Why not public datasets (Y2)
- Public SRE datasets (e.g., Google SLO/NFLX) exist but lack rich correlated logs + controllable
  remediation targets and per-service multi-signal telemetry.