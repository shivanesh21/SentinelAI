from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.recovery import RecoveryVerifier, verify_after_remediation
from src.remediation import MockBackend, RemediationExecutor, plan_for_category

SERVICE = "payment-api"
KEYS = ["Error_rate", "Latency_15min_avg", "Memory_growth_rate"]


def frame(rows):
    return pd.DataFrame(rows)


def make_rows(base, values, step_min=1):
    out = []
    for i, (err, lat, mem) in enumerate(values):
        out.append(
            {
                "service": SERVICE,
                "Error_rate": err,
                "Latency_15min_avg": lat,
                "Memory_growth_rate": mem,
                "timestamp": base + timedelta(minutes=i * step_min),
            }
        )
    return out


def show(report, label):
    print(f"\n--- {label} ---")
    print(f"  {report.action} on {report.service}: status={report.status} "
          f"recovery_time={report.recovery_time_sec:g}s (cooldown={report.cooldown_sec}s)")
    print(f"{'metric':22s} {'before':>10s} {'after':>10s} {'delta%':>10s} {'thr':>10s} {'status':>10s} rec")
    for m in report.metrics:
        print(f"{m.name:22s} {m.before:10.4g} {m.after:10.4g} {m.delta_pct:10.3g} "
              f"{str(m.threshold):>10s} {m.status:>10s} {m.recovered!s}")
    if report.log_path:
        print(f"  logged -> {report.log_path}")


def main() -> None:
    verifier = RecoveryVerifier(cooldown_sec=60)
    base = datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc)

    before_incident = make_rows(base, [(0.045, 248.0, 0.078), (0.05, 257.0, 0.085)])
    after_recovery = make_rows(base + timedelta(minutes=45), [(0.006, 98.0, 0.004), (0.005, 95.0, 0.003)])
    after_still_bad = make_rows(base + timedelta(minutes=45), [(0.048, 260.0, 0.082), (0.051, 268.0, 0.088)])

    print("=" * 78)
    print("RECOVERY VERIFICATION (Module 8) — automated re-check after remediation")
    print("=" * 78)

    report1 = verifier.verify(frame(before_incident), frame(after_recovery), SERVICE, action="restart_container")
    verifier.log(report1)
    show(report1, "CASE 1: restart fixed the memory leak")
    print("  -> SUCCESS as expected")

    report2 = verifier.verify(frame(before_incident), frame(after_still_bad), SERVICE, action="restart_container")
    verifier.log(report2)
    show(report2, "CASE 2: restart failed, metrics still elevated")
    print("  -> FAILURE as expected")

    print("\n" + "=" * 78)
    print("AUTOMATIC RE-CHECK AFTER A REMEDIATION ACTION")
    print("=" * 78)
    executor = RemediationExecutor(backend=MockBackend(), mode="approval", audit_path=None)
    plan = plan_for_category("memory_leak", SERVICE, context={"current_replicas": 1})
    execution = executor.execute(plan, approve=True)
    print(f"executed {execution.executed} action(s) in approval mode")
    after = verify_after_remediation(execution, frame(before_incident), frame(after_recovery), SERVICE,
                                     action=", ".join(r.name for r in execution.results if r.status == "ok"))
    verifier.log(after)
    show(after, "verify_after_remediation (auto re-check)")

    print("\n" + "=" * 78)
    print("SAMPLE LOGGED REPORT (data/recovery_log.jsonl)")
    print("=" * 78)
    for rec in RecoveryVerifier().read(limit=5):
        print(f"  {rec['checked_at'][:19]} {rec['service']} {rec['status']:8s} "
              f"recovery_time={rec['recovery_time_sec']:g}s action={rec['action']}")


if __name__ == "__main__":
    main()