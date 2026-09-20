# SentinelAI — Telemetry Collection Pipeline

Deliverable for Day 3 (Module 1). Turns the Day 2 simulator into a continuous
collection pipeline with time-partitioned storage.

## Architecture

```
Simulation tick (every N s)
      |
      +-- TelemetryGenerator.iter_steps()   (streaming engine, one tick at a time)
      |        | metrics + logs + incident events
      v        v
 TelemetryCollector
      |        buffers / appends
      v
 PartitionedStore
      +-- data/raw/metrics/dt=YYYY-MM-DD-HH/ ...   (hourly partitions)
      +-- data/raw/logs/dt=YYYY-MM-DD-HH/ ...
      +-- data/raw/truth/dt=YYYY-MM-DD-HH/ ...     (ground truth incidents)
```

## Components

| Component | Path | Responsibility |
|---|---|---|
| Streaming engine | `src/telemetry/simulator/generator.py` → `iter_steps()` | Yields `TickBatch` (metrics + logs + incident events) one tick at a time; shared by batch CLI and collector so both sinks consume identical data |
| Log generator | `src/telemetry/log_generator.py` | Lightweight INFO/WARNING/ERROR ambient lines (`heartbeat ok`, `slow query`, `connection retry`, `upstream timeout`); failure-correlated lines still come from scenario effects |
| Partitioned store | `src/telemetry/storage.py` → `PartitionedStore` | Time-series-friendly hourly partitions; CSV (append) or Parquet (rolling parts) |
| Collector | `src/telemetry/collector.py` → `TelemetryCollector` | Scheduled collection, live/continuous mode, open-incident tracking, clean stop |
| CLI | `scripts/collect_telemetry.py` | Manual operation / daemon |

## Usage

```bash
conda activate sentinelai

# Fixed window (batches of simulated time)
python scripts/collect_telemetry.py --duration-min 60 --format csv
python scripts/collect_telemetry.py --duration-min 60 --format parquet

# Continuous ("live") collection until Ctrl+C
python scripts/collect_telemetry.py            # uses config defaults (30 s interval)
python scripts/collect_telemetry.py --interval-sec 5

# Batch one-shot CSV/JSONL (Day 2 entry point, still available)
python scripts/generate_data.py --duration-min 720 --format csv
```

## Storage layout (read path)

- **CSV**: `data/raw/{metrics,logs,truth}/dt=<hour>/data.csv` — one header per file, appended across ticks.
- **Parquet**: `data/raw/{metrics,logs,truth}/dt=<hour>/part-<seq>.parquet` — rolled parts (default every 64 rows) by partition.
- Read with `glob("data/raw/metrics/dt=*/data.csv")` / `glob("data/raw/metrics/dt=*/*.parquet")`; pandas `concat` yields the full series (verified: 2 100 rows × 15 cols round-trip).

## Ground truth flow
- Incident `start` recorded when a scenario window opens; `end` when it closes.
- On clean stop/finish, any still-open incident is closed (end = last seen time) and flushed, so downstream labeling is never missing a window.

## Verification (performed)
- Batch generator regression: 5 min run → 50 rows / 0 incidents; 12 h run → 7 200 rows / 5 incidents.
- Collector CSV, 200 min → 2000 rows, 749 logs, 1 incident (memory leak, 120 min window), hourly partitions.
- Collector Parquet, 210 min → rolling parts + read-back sanity.
- Live mode: collects in real time at configured interval, stops cleanly on stop-event/Ctrl+C.