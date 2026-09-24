"""Day 24 stress runner: N services degrading simultaneously, end-to-end.

Every service gets its own incident (scenario types are cycled), all emitted
in one telemetry stream, so detection/remediation/recovery must stay isolated
per service. Fails if any service's recovery check is not SUCCESS.

Usage (from backend/):
    python scripts/stress_services.py --count 5 --mode approval
    python scripts/stress_services.py --services payment-api auth-service --scenario connection_pool_exhaustion
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from src.pipeline.e2e import SCENARIO_TYPES, PipelineEngine, build_services, configured_scenarios  # noqa: E402
from src.telemetry.simulator.failure_scenarios import ScenarioSpec  # noqa: E402
from src.telemetry.simulator.generator import TelemetryGenerator  # noqa: E402


def generate_concurrent_incidents(names: list[str], scenario_assignment: dict[str, str], duration_min: float, recovery_min: float, interval_sec: int, seed: int) -> dict:
    services = build_services(names, BACKEND / "config" / "services.yaml")
    start = datetime.now(timezone.utc).replace(microsecond=0)
    scenarios = [
        ScenarioSpec(service=name, scenario_type=scenario_type, start_offset_min=0.0, duration_min=duration_min)
        for name, scenario_type in scenario_assignment.items()
    ]
    before_rows: list[dict] = []
    generator = TelemetryGenerator(interval_sec=interval_sec, seed=seed, start_ts=start)
    for batch in generator.iter_steps(services, scenarios, duration_min=duration_min):
        for metric in batch.metric_rows:
            before_rows.append(metric.to_dict())

    after_rows: list[dict] = []
    second_start = start + timedelta(minutes=duration_min)
    generator2 = TelemetryGenerator(interval_sec=interval_sec, seed=seed + 101, start_ts=second_start)
    for batch in generator2.iter_steps(services, [], duration_min=recovery_min):
        for metric in batch.metric_rows:
            after_rows.append(metric.to_dict())

    return {"services": [s.name for s in services], "before_rows": before_rows, "after_rows": after_rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int)
    parser.add_argument("--services", nargs="*")
    parser.add_argument("--scenario", choices=SCENARIO_TYPES)
    parser.add_argument("--mode", choices=["advisory", "approval", "autonomous"])
    parser.add_argument("--duration-min", type=float, default=40)
    parser.add_argument("--recovery-min", type=float, default=20)
    parser.add_argument("--interval-sec", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    configured = configured_scenarios(BACKEND / "config" / "services.yaml")["services"]
    names = args.services or configured
    if args.count is not None:
        if args.services:
            parser.error("pass either --services or --count, not both")
        names = configured[: args.count]
    if not names:
        parser.error("no services to stress (are --services valid?)")

    assignment = {name: (args.scenario or SCENARIO_TYPES[i % len(SCENARIO_TYPES)]) for i, name in enumerate(names)}
    print(f"services: {names}")
    print(f"scenario per service: {assignment}")

    incident = generate_concurrent_incidents(names, assignment, args.duration_min, args.recovery_min, args.interval_sec, args.seed)
    total_rows = len(incident["before_rows"]) + len(incident["after_rows"])

    engine = PipelineEngine(BACKEND, mode=args.mode)
    started = time.perf_counter()
    outcome = engine.run(
        incident["before_rows"],
        mode=args.mode,
        per_service=True,
        recovery=(incident["before_rows"], incident["after_rows"]),
    )
    wall = time.perf_counter() - started

    detection = outcome["detection"]
    described = {row["service"]: row for row in detection["latest"]}
    services_seen = sorted(described)
    remediation = outcome.get("remediation") or {}
    remediated = {o["service"]: o for o in remediation.get("outcomes") or []}
    checks = {(c.get("service"), c.get("status")) for c in (outcome.get("recovery") or {}).get("checks") or []}
    check_by_service = {c["service"]: c for c in (outcome.get("recovery") or {}).get("checks") or []}

    print("\nper-service report:")
    clean = True
    for svc in names:
        row = described.get(svc)
        recovered = check_by_service.get(svc, {}).get("status")
        state = "OK" if (row and svc in remediated and recovered == "SUCCESS") else "FAIL"
        clean = clean and state == "OK"
        print(f"  {state:4s} {svc:22s} risk={row['risk_level'] if row else '-'} "
              f"score={row['risk_score'] if row else '-'} category={remediated.get(svc, {}).get('category', '-')} "
              f"recovery={recovered}")

    summary = {
        "services": names,
        "scenario_per_service": [assignment[n] for n in names],
        "rows_total": total_rows,
        "rows_scored": outcome["rows"],
        "n_flagged": detection["n_flagged"],
        "n_services_detected": detection["n_services"],
        "n_services_expected": len(names),
        "mode": outcome["mode"],
        "executed_actions": remediation.get("executed", 0),
        "recovery_states": [check_by_service.get(svc, {}).get("status") for svc in names],
        "wall_sec": round(wall, 2),
        "rows_per_sec": round(total_rows / wall, 1),
        "timings_ms": outcome["timings_ms"],
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(outcome, indent=2, default=str))

    failure = services_seen != names or detection["n_services"] != len(names)
    missing = [s for s in names if s not in remediated]
    if failure or missing or not clean:
        print(f"\n[FAIL] isolation broken: services detected={services_seen} un-remediated={missing}")
        return 1
    print(f"\n[PASS] all {len(names)} services isolated end-to-end in {round(wall, 2)}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())