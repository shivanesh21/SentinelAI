from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rca import RCAClient, build_rca_prompt


def evidence_package(
    service: str,
    risk_level: str,
    risk_score: float,
    failure_probability: float,
    components: dict,
    features: list[dict],
    deltas: dict,
    logs: list[dict],
) -> dict:
    return {
        "timestamp": "2026-09-19T04:30:08+00:00",
        "service": service,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "failure_probability": failure_probability,
        "components": components,
        "top_anomalous_features": features,
        "metrics_deltas": deltas,
        "correlated_logs": logs,
        "recommended_actions": [],
    }


SCENARIOS = [
    {
        "name": "DB pool exhaustion",
        "evidence": evidence_package(
            service="payment-api",
            risk_level="CRITICAL",
            risk_score=0.92,
            failure_probability=0.87,
            components={"failure_probability": 0.87, "error_rate": 0.90, "latency_trend": 0.80, "memory_growth": 0.30},
            features=[
                {"feature": "Request_rate", "current": 4200.0, "baseline_mean": 1800.0, "zscore": 4.5, "direction": "up"},
                {"feature": "DB_usage_5min_avg", "current": 88.0, "baseline_mean": 61.0, "zscore": 3.8, "direction": "up"},
                {"feature": "Error_rate", "current": 9.2, "baseline_mean": 1.1, "zscore": 3.4, "direction": "up"},
            ],
            deltas={"Error_rate": 736.0, "Latency_15min_avg": 210.0, "Request_rate": 133.0},
            logs=[
                {"level": "ERROR", "message": "[payment-api] Connection pool exhausted - 0 available (max 50) after 30s acquire timeout"},
                {"level": "WARN", "message": "[payment-api] JDBC connection borrow timed out; queuing request"},
            ],
        ),
    },
    {
        "name": "memory leak",
        "evidence": evidence_package(
            service="order-service",
            risk_level="CRITICAL",
            risk_score=0.95,
            failure_probability=0.93,
            components={"failure_probability": 0.93, "error_rate": 0.40, "latency_trend": 0.35, "memory_growth": 0.98},
            features=[
                {"feature": "Memory_growth_rate", "current": 0.048, "baseline_mean": 0.001, "zscore": 6.1, "direction": "up"},
                {"feature": "Memory_15min_avg", "current": 94.2, "baseline_mean": 68.5, "zscore": 5.2, "direction": "up"},
            ],
            deltas={"Memory_growth_rate": 4800.0, "CPU_15min_avg": 31.0},
            logs=[
                {"level": "WARN", "message": "[order-service] Heap usage > 90% - GC pause 1.4s detected"},
                {"level": "ERROR", "message": "[order-service] OutOfMemoryError in cache eviction thread"},
            ],
        ),
    },
    {
        "name": "latency spike",
        "evidence": evidence_package(
            service="auth-service",
            risk_level="ANOMALOUS",
            risk_score=0.78,
            failure_probability=0.71,
            components={"failure_probability": 0.71, "error_rate": 0.5, "latency_trend": 0.95, "memory_growth": 0.2},
            features=[
                {"feature": "Latency_5min_avg", "current": 980.0, "baseline_mean": 210.0, "zscore": 5.8, "direction": "up"},
                {"feature": "latency_per_request", "current": 1.1, "baseline_mean": 0.21, "zscore": 4.9, "direction": "up"},
            ],
            deltas={"Latency_15min_avg": 366.0, "Request_rate": 12.0},
            logs=[
                {"level": "WARN", "message": "[auth-service] p99 latency 1.1s exceeded SLO of 500ms"},
                {"level": "WARN", "message": "[auth-service] Queue depth growing - 3400 requests buffered"},
            ],
        ),
    },
    {
        "name": "network partition",
        "evidence": evidence_package(
            service="notification-worker",
            risk_level="CRITICAL",
            risk_score=0.88,
            failure_probability=0.82,
            components={"failure_probability": 0.82, "error_rate": 0.86, "latency_trend": 0.60, "memory_growth": 0.15},
            features=[
                {"feature": "Network_in_5min_avg", "current": 4.2, "baseline_mean": 55.0, "zscore": -6.4, "direction": "down"},
                {"feature": "Network_in_15min_avg", "current": 6.0, "baseline_mean": 57.0, "zscore": -5.9, "direction": "down"},
                {"feature": "Error_rate", "current": 15.0, "baseline_mean": 0.9, "zscore": 4.2, "direction": "up"},
            ],
            deltas={"Error_rate": 1566.0, "Latency_15min_avg": 88.0},
            logs=[
                {"level": "ERROR", "message": "[notification-worker] Connection refused to broker.platform:9092"},
                {"level": "ERROR", "message": "[notification-worker] Kafka partition leader unreachable - retrying in 3s"},
            ],
        ),
    },
    {
        "name": "disk fill",
        "evidence": evidence_package(
            service="inventory-service",
            risk_level="ANOMALOUS",
            risk_score=0.75,
            failure_probability=0.68,
            components={"failure_probability": 0.68, "error_rate": 0.55, "latency_trend": 0.45, "memory_growth": 0.25},
            features=[
                {"feature": "Disk_growth_rate", "current": 0.061, "baseline_mean": 0.002, "zscore": 5.7, "direction": "up"},
                {"feature": "Disk_15min_avg", "current": 96.4, "baseline_mean": 72.0, "zscore": 4.3, "direction": "up"},
            ],
            deltas={"Disk_growth_rate": 2950.0, "Error_rate": 260.0},
            logs=[
                {"level": "ERROR", "message": "[inventory-service] write error: No space left on device on /var/lib/mysql"},
                {"level": "WARN", "message": "[inventory-service] Filesystem 95% full - log rotation pending"},
            ],
        ),
    },
    {
        "name": "code deployment regression",
        "evidence": evidence_package(
            service="payment-api",
            risk_level="WARNING",
            risk_score=0.55,
            failure_probability=0.48,
            components={"failure_probability": 0.48, "error_rate": 0.45, "latency_trend": 0.30, "memory_growth": 0.10},
            features=[
                {"feature": "Error_rate", "current": 2.4, "baseline_mean": 0.8, "zscore": 2.6, "direction": "up"},
                {"feature": "Layout_10min_avg", "current": 140.0, "baseline_mean": 120.0, "zscore": 2.1, "direction": "up"},
            ],
            deltas={"Error_rate": 200.0, "Request_rate": 4.0},
            logs=[
                {"level": "WARN", "message": "[payment-api] New deployment started - v2.1.4 rolling out"},
                {"level": "WARN", "message": "[payment-api] Elevated 5xx errors after rollout started"},
            ],
        ),
    },
]


def main() -> None:
    client = RCAClient()
    print(f"RCA provider: {client.provider}")
    print("=" * 78)

    for i, scenario in enumerate(SCENARIOS, start=1):
        evidence = scenario["evidence"]

        print(f"\n{'='*78}")
        print(f"SCENARIO #{i}: {scenario['name'].upper()}  (service={evidence['service']})")
        print(f"{'='*78}")

        system, user = build_rca_prompt(evidence)
        print(f"\n-- PROMPT (user) --")
        print(user[:400])

        result = client.analyze(evidence)
        print(f"\n-- RESULT --")
        print(result.to_json())


if __name__ == "__main__":
    main()