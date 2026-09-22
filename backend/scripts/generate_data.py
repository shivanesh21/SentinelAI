from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.telemetry.config import (
    default_duration,
    load_scenarios,
    load_services,
    load_settings,
)
from src.telemetry.simulator.generator import TelemetryGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic SentinelAI telemetry")
    parser.add_argument("--duration-min", type=float, default=None)
    parser.add_argument("--interval-sec", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    parser.add_argument("--format", choices=["csv", "jsonl"], default="csv")
    parser.add_argument("--services", type=Path, default=ROOT / "config" / "services.yaml")
    parser.add_argument("--settings", type=Path, default=ROOT / "config" / "settings.yaml")
    parser.add_argument("--start-ts", type=str, default=None)
    args = parser.parse_args()

    settings = load_settings(args.settings)
    telemetry_cfg = settings.get("telemetry", {})
    simulation_cfg = settings.get("simulation", {})

    duration_min = args.duration_min if args.duration_min is not None else default_duration(args.settings)
    interval_sec = args.interval_sec if args.interval_sec is not None else int(telemetry_cfg.get("collection_interval_sec", 30))
    seed = args.seed if args.seed is not None else int(simulation_cfg.get("seed", 42))
    start_ts = datetime.fromisoformat(args.start_ts) if args.start_ts else datetime.now(timezone.utc).replace(microsecond=0)

    services = load_services(args.services)
    scenarios = load_scenarios(args.settings)

    generator = TelemetryGenerator(interval_sec=interval_sec, seed=seed, start_ts=start_ts)
    counts = generator.run(services, scenarios, duration_min, out_dir=args.out, output_format=args.format)
    print(f"done: {counts['rows']} metric rows, {counts['logs']} log lines, {counts['incidents']} incidents "
          f"(format={args.format}, duration={duration_min:.0f}min)")


if __name__ == "__main__":
    main()