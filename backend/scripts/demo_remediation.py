from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.remediation import (
    MockBackend,
    RemediationExecutor,
    RemediationPlanner,
    resolve_mode,
)


def main() -> None:
    categories = [
        "memory_leak",
        "connection_pool_exhaustion",
        "latency_spike",
        "network_partition",
        "disk_fill",
        "code_deployment",
        "traffic_spike",
    ]
    services = {
        "memory_leak": "order-service",
        "connection_pool_exhaustion": "payment-api",
        "latency_spike": "auth-service",
        "network_partition": "notification-worker",
        "disk_fill": "inventory-service",
        "code_deployment": "payment-api",
        "traffic_spike": "web-gateway",
    }

    planner = RemediationPlanner()
    backend = MockBackend()
    mode = resolve_mode()
    print(f"Remediation mode: {mode}")
    print("=" * 78)

    for category in categories:
        service = services[category]
        plan = planner.plan_for_category(category, service, context={"current_replicas": 1})
        print(f"\n{category.upper()} ({service}) -> {len(plan)} action(s)")
        for spec in plan:
            print(f"  - {spec.name:26s} reversible={spec.reversible!s:5s} risk={spec.risk:6s} :: {spec.description}")

    print("\n" + "=" * 78)
    print("PLAN EXECUTION (approval mode -> simulated)")
    print("=" * 78)
    executor = RemediationExecutor(backend=backend, mode=mode, audit_path=None)
    plan = planner.plan_for_category("traffic_spike", "web-gateway", context={"current_replicas": 2})
    report = executor.execute(plan)
    print(f"mode={report.mode} dry_run={report.dry_run} executed={report.executed}")
    for result in report.results:
        print(f"  [{result.status:6s}] {result.name}: {result.detail}")

    print("\n" + "=" * 78)
    print("PLAN EXECUTION (approved -> real against MockBackend)")
    print("=" * 78)
    approved = RemediationExecutor(backend=backend, mode="approval", audit_path=None).execute(plan, approve=True)
    print(f"mode={approved.mode} dry_run={approved.dry_run} executed={approved.executed}")
    for result in approved.results:
        print(f"  [{result.status:6s}] {result.name}: {result.detail}")

    print("\n" + "=" * 78)
    print("ROLLBACK of approved execution")
    print("=" * 78)
    rolled = RemediationExecutor(backend=backend, mode="approval", audit_path=None).rollback(approved)
    for result in rolled.results:
        print(f"  [{result.status}] {result.name}: {result.detail}")

    print("\naudit trail (executor with audit_path):")
    audited = RemediationExecutor(backend=MockBackend(), mode="approval").execute(plan)
    print(f"  wrote {audited.audit_path}")


if __name__ == "__main__":
    main()