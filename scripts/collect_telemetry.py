from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.telemetry.collector import TelemetryCollector, build_store
from src.telemetry.config import (
    default_duration,
    load_scenarios,
    load_services,
    load_settings,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuously collect SentinelAI telemetry")
    parser.add_argument("--duration-min", type=float, default=None, help="run fixed duration; omit for continuous")
    parser.add_argument("--interval-sec", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv")
    parser.add_argument("--start-ts", type=str, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    parser.add_argument("--services", type=Path, default=ROOT / "config" / "services.yaml")
    parser.add_argument("--settings", type=Path, default=ROOT / "config" / "settings.yaml")
    args = parser.parse_args()

    settings = load_settings(args.settings)
    telemetry_cfg = settings.get("telemetry", {})
    simulation_cfg = settings.get("simulation", {})

    interval_sec = args.interval_sec if args.interval_sec is not None else int(telemetry_cfg.get("collection_interval_sec", 30))
    seed = args.seed if args.seed is not None else int(simulation_cfg.get("seed", 42))
    start_ts = datetime.fromisoformat(args.start_ts) if args.start_ts else datetime.now(timezone.utc).replace(microsecond=0)
    duration_min = args.duration_min

    services = load_services(args.services)
    scenarios = load_scenarios(args.settings)
    store = build_store(args.out, fmt=args.format)
    collector = TelemetryCollector(
        store=store,
        interval_sec=interval_sec,
        seed=seed,
        start_ts=start_ts,
        live=duration_min is None,
    )

    print(f"SENTINELAI COLLECTOR  interval={interval_sec}s  format={args.format}  "
          f"services={len(services)}  scenarios={len(scenarios)}")
    if duration_min is None:
        print("running continuously ... press Ctrl+C to stop")
    try:
        totals = collector.collect(services, scenarios, duration_min)
        print(f"done: {totals['rows']} metric rows, {totals['logs']} log lines, "
              f"{totals['incidents']} incidents -> {store.base_dir}")
    finally:
        store.close()


if __name__ == "__main__":
    main()