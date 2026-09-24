from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.pipeline.e2e import (
    AnomalyAdapter,
    PipelineEngine,
    generate_incident,
    latest_per_service,
)
from src.telemetry.simulator.failure_scenarios import ScenarioSpec
from src.telemetry.simulator.generator import TelemetryGenerator
from src.pipeline.e2e import build_services

BACKEND = ROOT
MODELS = ROOT / "models"


def _concurrent_incidents(names, assignments, duration_min=25, recovery_min=15, interval_sec=30, seed=3):
    services = build_services(names, BACKEND / "config" / "services.yaml")
    start = datetime.now(timezone.utc).replace(microsecond=0)
    scenarios = [
        ScenarioSpec(service=name, scenario_type=scenario_type, start_offset_min=0.0, duration_min=duration_min)
        for name, scenario_type in assignments.items()
    ]
    before_rows = []
    for batch in TelemetryGenerator(interval_sec=interval_sec, seed=seed, start_ts=start).iter_steps(
        services, scenarios, duration_min=duration_min
    ):
        for metric in batch.metric_rows:
            before_rows.append(metric.to_dict())
    after_rows = []
    second_start = start + timedelta(minutes=duration_min)
    for batch in TelemetryGenerator(interval_sec=interval_sec, seed=seed + 101, start_ts=second_start).iter_steps(
        services, [], duration_min=recovery_min
    ):
        for metric in batch.metric_rows:
            after_rows.append(metric.to_dict())
    return before_rows, after_rows


def _models_present():
    return (
        (MODELS / "baseline" / "logistic_regression").exists()
        and (MODELS / "risk" / "calibration.json").exists()
    )


class PipelineEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _models_present():
            raise unittest.SkipTest("trained models not present")
        cls.engine = PipelineEngine(BACKEND)

    def test_anomaly_adapter_contract(self):
        adapter = self.engine.anomaly_detector()
        self.assertIsInstance(adapter, AnomalyAdapter)
        incident = generate_incident(
            ["payment-api"], "memory_leak", duration_min=25, recovery_min=10, interval_sec=30, seed=1,
            services_path=BACKEND / "config" / "services.yaml",
        )
        features = self.engine.features_from_raw(incident["before_rows"])
        scored = adapter.score(features)
        column = f"{adapter.primary}_score"
        self.assertIn(column, scored.columns)
        self.assertEqual(len(scored), len(features))
        values = scored[column].dropna().to_numpy()
        self.assertTrue(np.all((values >= 0.0) & (values <= 1.0)))
        self.assertEqual(scored["service"].tolist(), features["service"].tolist())

    def test_full_chain_incident_recovers(self):
        incident = generate_incident(
            ["payment-api"], "latency_spike", duration_min=25, recovery_min=15, interval_sec=30, seed=2,
            services_path=BACKEND / "config" / "services.yaml",
        )
        outcome = self.engine.run(
            incident["before_rows"],
            mode="advisory",
            per_service=True,
            recovery=(incident["before_rows"], incident["after_rows"]),
        )
        self.assertGreater(outcome["rows"], 0)
        self.assertGreaterEqual(outcome["detection"]["n_flagged"], 1)
        latest = outcome["detection"]["latest"]
        self.assertEqual([row["service"] for row in latest], ["payment-api"])
        self.assertTrue(all(row["risk"] == 1 for row in latest))
        self.assertTrue(any(row["anomaly_score"] >= 0.0 for row in latest))
        self.assertIsNotNone(outcome["remediation"])
        self.assertTrue(outcome["remediation"]["analyzed"])
        checks = (outcome["recovery"] or {}).get("checks", [])
        self.assertEqual([c["status"] for c in checks], ["SUCCESS"])
        for stage in ("features_then_scoring", "evidence", "remediation", "recovery", "total"):
            self.assertGreater(outcome["timings_ms"][stage], 0.0)

    def test_healthy_telemetry_raises_no_incident(self):
        incident = generate_incident(
            ["payment-api"], None, duration_min=20, recovery_min=10, interval_sec=30, seed=4,
            services_path=BACKEND / "config" / "services.yaml",
        )
        outcome = self.engine.run(incident["before_rows"], mode="advisory", per_service=True)
        self.assertEqual(outcome["detection"]["n_flagged"], 0)
        self.assertIsNone(outcome["remediation"])
        self.assertIsNone(outcome["recovery"])

    def test_multi_service_per_service_isolation(self):
        names = ["payment-api", "order-service", "auth-service"]
        assignments = {
            "payment-api": "memory_leak",
            "order-service": "latency_spike",
            "auth-service": "connection_pool_exhaustion",
        }
        before_rows, after_rows = _concurrent_incidents(names, assignments)
        outcome = self.engine.run(
            before_rows,
            mode="advisory",
            per_service=True,
            recovery=(before_rows, after_rows),
        )
        latest = {row["service"]: row for row in outcome["detection"]["latest"]}
        self.assertEqual(sorted(latest), sorted(names))
        self.assertEqual(outcome["detection"]["n_services"], len(names))
        self.assertTrue(all(row["risk"] == 1 for row in latest.values()))
        remediated = [o["service"] for o in (outcome["remediation"] or {}).get("outcomes", [])]
        self.assertEqual(sorted(remediated), sorted(names))
        records = outcome["remediation"]["outcomes"]
        categories = {r["service"]: r["category"] for r in records}
        for service in names:
            self.assertIsNotNone(categories[service])
        checks = {c["service"]: c["status"] for c in (outcome["recovery"] or {}).get("checks", [])}
        self.assertEqual(sorted(checks), sorted(names))
        self.assertTrue(all(status == "SUCCESS" for status in checks.values()))

    def test_latest_per_service_ranks_by_risk(self):
        incident = generate_incident(
            ["payment-api", "order-service"], "memory_leak", duration_min=20, recovery_min=10, interval_sec=30, seed=5,
            services_path=BACKEND / "config" / "services.yaml",
        )
        features = self.engine.features_from_raw(incident["before_rows"])
        scored = self.engine.risk_engine().score(features)
        latest = latest_per_service(scored)
        self.assertEqual(set(latest["service"]), {"payment-api", "order-service"})
        self.assertTrue((latest["risk_score"].values == np.sort(latest["risk_score"].values)[::-1]).all())


if __name__ == "__main__":
    unittest.main()