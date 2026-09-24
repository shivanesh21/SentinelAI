"""Day 24 runner: exercise one or every incident scenario end-to-end.

Usage (from backend/):
    python scripts/run_incident.py --scenario latency_spike --mode approval [--service payment-api]
    python scripts/run_incident.py --all --mode autonomous --duration-min 30
    python scripts/run_incident.py --scenario disk_fill --service notification-worker

Flags:
    --scenario NAME      one of the five simulated scenario types
    --all                sweep every scenario type
    --service NAME       restrict the incident to one configured service
    --mode MODE          advisory | approval | autonomous (or leave the default)
    --duration-min N     incident duration in minutes (default 45)
    --recovery-min N     healthy tail used for the recovery check (default 20)
    --interval-sec N     telemetry cadence in seconds (default 30)
    --seed N             RNG seed for reproducible telemetry (default 42)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from src.pipeline.e2e import (  # noqa: E402
    SCENARIO_TYPES,
    PipelineEngine,
    configured_scenarios,
    generate_incident,
)


def run_one(engine: PipelineEngine, scenario: str, service: str | None, mode: str | None, args: argparse.Namespace) -> dict:
    names = [service] if service else configured_scenarios(BACKEND / "config" / "services.yaml")["services"]
    incident = generate_incident(
        names,
        scenario,
        duration_min=args.duration_min,
        recovery_min=args.recovery_min,
        interval_sec=args.interval_sec,
        seed=args.seed,
        services_path=BACKEND / "config" / "services.yaml",
    )
    started = time.perf_counter()
    outcome = engine.run(
        incident["before_rows"],
        mode=mode,
        per_service=True,
        recovery=(incident["before_rows"], incident["after_rows"]),
    )
    wall = time.perf_counter() - started
    outcome["incident"] = {
        "scenario_type": incident["scenario_type"],
        "target_service": incident["service"],
        "interval_sec": incident["interval_sec"],
        "start_ts": incident["start_ts"],
    }
    recovery_checks = (outcome.get("recovery") or {}).get("checks") or []
    summary = {
        "scenario": scenario,
        "services": outcome["services"],
        "rows": outcome["rows"],
        "flagged": outcome["detection"]["n_flagged"],
        "n_services": outcome["detection"]["n_services"],
        "mode": outcome["mode"],
        "executed_actions": (outcome.get("remediation") or {}).get("executed", 0),
        "recovery_states": [c.get("status") for c in recovery_checks],
        "wall_sec": round(wall, 2),
    }
    if args.verbose:
        print(json.dumps(summary, indent=2))
    else:
        print("  scenario={scenario} mode={mode} flagged={flagged} executed={executed_actions} "
              "recovery={recovery_states} wall={wall_sec}s".format(**summary))
    return outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", choices=SCENARIO_TYPES)
    parser.add_argument("--all", action="store_true", help="sweep every scenario type")
    parser.add_argument("--service")
    parser.add_argument("--mode", choices=["advisory", "approval", "autonomous"])
    parser.add_argument("--duration-min", type=float, default=45)
    parser.add_argument("--recovery-min", type=float, default=20)
    parser.add_argument("--interval-sec", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out", type=Path, default=None, help="write full JSON per scenario to this dir")
    args = parser.parse_args(argv)

    if bool(args.scenario) == bool(args.all):
        parser.error("pass exactly one of --scenario NAME or --all")
    if args.all and args.service:
        parser.error("--service cannot be used with --all")

    engine = PipelineEngine(BACKEND, mode=args.mode)
    targets = SCENARIO_TYPES if args.all else [args.scenario]

    print(f"engine readiness: {engine.readiness()['ready']} (mode {engine.readiness()['mode']})")
    overall = {"ok": 0, "total": 0}
    for scenario in targets:
        outcome = run_one(engine, scenario, args.service, args.mode, args)
        overall["total"] += 1
        states = [c.get("status") for c in (outcome.get("recovery") or {}).get("checks") or []]
        if outcome["rows"] > 0 and (not states or all(s == "SUCCESS" for s in states)):
            overall["ok"] += 1
            print(f"  [PASS] {scenario}")
        else:
            print(f"  [FAIL] {scenario} recovery_states={states}")
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / f"{scenario.replace('_', '-')}.json").write_text(json.dumps(outcome, indent=2, default=str))

    print(f"\n{overall['ok']}/{overall['total']} scenarios fully recovered")
    return 0 if overall["ok"] == overall["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())