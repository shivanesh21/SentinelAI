from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.risk import RiskEngine, format_risk
from src.telemetry.config import load_settings


def main() -> None:
    settings = load_settings(ROOT / "config" / "settings.yaml")
    engine = RiskEngine.load(ROOT / "models", settings)

    # Load test data to show real examples
    from src.features.store import FeatureStore
    store = FeatureStore(ROOT / "data" / "features")
    test = store.load_split("test", ROOT / "data" / "splits")

    # Score the test split
    scored = engine.score(test)

    print("=" * 70)
    print("RISK SCORING ENGINE — EXAMPLE OUTPUTS (test split)")
    print("=" * 70)

    # Show top 10 riskiest timestamps across all services
    top = scored.nlargest(10, "risk_score")[
        ["timestamp", "service", "risk_score", "risk_level", "failure_probability", "error_rate", "latency_trend", "memory_growth"]
    ]
    print("\nTOP 10 RISKIEST TIMESTAMPS:")
    for _, row in top.iterrows():
        line = format_risk(row["risk_score"], engine.severity_thresholds)
        print(f"  {row['timestamp']} | {row['service']:20s} | {line}")
        print(f"    failure_prob={row['failure_probability']:.3f}  error_rate={row['error_rate']:.3f}  "
              f"latency_trend={row['latency_trend']:.3f}  memory_growth={row['memory_growth']:.3f}")

    # Show latest risk per service
    print("\n" + "=" * 70)
    print("LATEST RISK PER SERVICE (most recent timestamp):")
    print("=" * 70)
    latest = engine.latest_by_service(test)
    for _, row in latest.iterrows():
        line = format_risk(row["risk_score"], engine.severity_thresholds)
        print(f"  {row['service']:20s} | {line}")
        print(f"    failure_prob={row['failure_probability']:.3f}  error_rate={row['error_rate']:.3f}  "
              f"latency_trend={row['latency_trend']:.3f}  memory_growth={row['memory_growth']:.3f}")

    # Show severity distribution
    print("\n" + "=" * 70)
    print("SEVERITY DISTRIBUTION (test split):")
    print("=" * 70)
    dist = scored["risk_level"].value_counts()
    for level in ["NORMAL", "WARNING", "ANOMALOUS", "CRITICAL"]:
        count = dist.get(level, 0)
        pct = count / len(scored) * 100
        print(f"  {level:12s}: {count:5d} ({pct:.1f}%)")

    # Example: simulate a high-risk scenario
    print("\n" + "=" * 70)
    print("SIMULATED HIGH-RISK SCENARIO:")
    print("=" * 70)
    # Take a normal row and spike the signals
    sample = test.sample(1, random_state=42).copy()
    sample["Error_rate"] = sample["Error_rate"].max() * 3
    sample["Latency_15min_avg"] = sample["Latency_15min_avg"].max() * 2
    sample["Memory_growth_rate"] = sample["Memory_growth_rate"].max() * 3
    scored_sample = engine.score(sample)
    row = scored_sample.iloc[0]
    print(f"  {row['timestamp']} | {row['service']} | {format_risk(row['risk_score'], engine.severity_thresholds)}")
    print(f"    failure_prob={row['failure_probability']:.3f}  error_rate={row['error_rate']:.3f}  "
          f"latency_trend={row['latency_trend']:.3f}  memory_growth={row['memory_growth']:.3f}")


if __name__ == "__main__":
    main()