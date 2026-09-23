import os
import tempfile
import unittest
from datetime import timedelta

import pandas as pd

from src.recovery import (
    DEFAULT_COOLDOWN_SEC,
    DEFAULT_METRICS,
    MetricCheck,
    RecoveryReport,
    RecoveryVerifier,
    verify_after_remediation,
)
from src.remediation import ActionSpec, MockBackend, RemediationExecutor


def _frame(rows):
    return pd.DataFrame(rows)


def _row(service, error, latency, memory, ts=None):
    row = {
        "service": service,
        "Error_rate": error,
        "Latency_15min_avg": latency,
        "Memory_growth_rate": memory,
    }
    if ts is not None:
        row["timestamp"] = ts
    return row


UNHEALTHY = _row("payment-api", 0.04, 245.0, 0.075)
HEALTHY = _row("payment-api", 0.005, 95.0, 0.004)
WORSE = _row("payment-api", 0.07, 340.0, 0.09)
STILL_BAD = _row("payment-api", 0.041, 250.0, 0.076)


class TestSnapshot(unittest.TestCase):
    def setUp(self):
        self.verifier = RecoveryVerifier()

    def test_filters_by_service_and_averages(self):
        frame = _frame(
            [
                UNHEALTHY,
                _row("order-service", 0.002, 80.0, 0.001),
                HEALTHY,
            ]
        )
        snap = self.verifier.snapshot(frame, "payment-api")
        self.assertAlmostEqual(snap["Error_rate"], 0.0225)

    def test_unknown_service_raises(self):
        frame = _frame([UNHEALTHY, HEALTHY])
        with self.assertRaises(ValueError):
            self.verifier.snapshot(frame, "ghost-service")

    def test_missing_metric_columns_raises(self):
        frame = pd.DataFrame([{"service": "payment-api", "cpu_util": 1.0}])
        with self.assertRaises(ValueError):
            self.verifier.snapshot(frame, "payment-api")


class TestVerify(unittest.TestCase):
    def setUp(self):
        self.verifier = RecoveryVerifier(cooldown_sec=300)

    def test_success_when_metrics_recover(self):
        report = self.verifier.verify(
            _frame([UNHEALTHY]), _frame([HEALTHY]), "payment-api", action="clear_cache"
        )
        self.assertEqual(report.status, "SUCCESS")
        self.assertEqual(report.action, "clear_cache")
        self.assertEqual(report.recovery_time_sec, 300.0)
        self.assertEqual(len(report.metrics), len(DEFAULT_METRICS))
        self.assertAllRecovered(report)

    def test_failure_when_metrics_worsen(self):
        report = self.verifier.verify(_frame([UNHEALTHY]), _frame([WORSE]), "payment-api")
        self.assertEqual(report.status, "FAILURE")
        statuses = [m.status for m in report.metrics]
        self.assertIn("worsened", statuses)

    def test_failure_when_unhealthy_and_not_improving(self):
        report = self.verifier.verify(_frame([UNHEALTHY]), _frame([STILL_BAD]), "payment-api")
        self.assertEqual(report.status, "FAILURE")

    def test_improved_but_above_threshold_counts_as_recovered(self):
        improved = _row("payment-api", 0.03, 220.0, 0.06)
        report = self.verifier.verify(_frame([UNHEALTHY]), _frame([improved]), "payment-api")
        self.assertEqual(report.status, "SUCCESS")
        self.assertTrue(all(m.recovered for m in report.metrics))

    def test_recovery_time_uses_timestamps_when_available(self):
        base = pd.Timestamp("2026-09-22 12:00:00")
        before = _frame(
            [_row("payment-api", 0.04, 245.0, 0.075, ts=base), _row("payment-api", 0.045, 255.0, 0.08, ts=base + timedelta(minutes=5))]
        )
        after = _frame(
            [_row("payment-api", 0.005, 95.0, 0.004, ts=base + timedelta(minutes=40)), _row("payment-api", 0.006, 98.0, 0.005, ts=base + timedelta(minutes=45))]
        )
        report = self.verifier.verify(before, after, "payment-api")
        self.assertEqual(report.status, "SUCCESS")
        self.assertEqual(report.recovery_time_sec, 2400.0)

    def test_recovery_time_floored_to_cooldown(self):
        base = pd.Timestamp("2026-09-22 12:00:00")
        before = _frame([_row("payment-api", 0.04, 245.0, 0.075, ts=base)])
        after = _frame([_row("payment-api", 0.005, 95.0, 0.004, ts=base + timedelta(seconds=10))])
        report = self.verifier.verify(before, after, "payment-api")
        self.assertEqual(report.recovery_time_sec, 300.0)

    def test_verify_snapshots_direct(self):
        report = self.verifier.verify_snapshots(
            {"Error_rate": 0.04, "Latency_15min_avg": 245.0, "Memory_growth_rate": 0.075},
            {"Error_rate": 0.005, "Latency_15min_avg": 95.0, "Memory_growth_rate": 0.004},
            "payment-api",
        )
        self.assertEqual(report.status, "SUCCESS")

    def assertAllRecovered(self, report):
        self.assertTrue(all(m.recovered for m in report.metrics), [m.status for m in report.metrics])


class TestLogging(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_log_and_read_round_trip(self):
        verifier = RecoveryVerifier(log_path=os.path.join(self.tmp.name, "recovery_log.jsonl"))
        report = verifier.verify(_frame([UNHEALTHY]), _frame([HEALTHY]), "payment-api")
        verifier.log(report)
        records = verifier.read()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "SUCCESS")
        self.assertEqual(records[0]["recovery_time_sec"], float(verifier.cooldown_sec))
        self.assertEqual(records[0]["metrics"][0]["name"], "Error_rate")

    def test_read_respects_limit(self):
        verifier = RecoveryVerifier(log_path=os.path.join(self.tmp.name, "recovery_log_limit.jsonl"))
        for i in range(3):
            verifier.log(verifier.verify_snapshots(
                {"Error_rate": 0.04, "Latency_15min_avg": 245.0, "Memory_growth_rate": 0.075},
                {"Error_rate": 0.005, "Latency_15min_avg": 95.0, "Memory_growth_rate": 0.004},
                f"svc-{i}",
            ))
        self.assertEqual(len(verifier.read(limit=2)), 2)
        self.assertEqual(verifier.read(0), [])


class TestVerifyAfterRemediation(unittest.TestCase):
    def test_skipped_when_nothing_executed(self):
        executor = RemediationExecutor(MockBackend(), mode="advisory")
        report = executor.execute([], approve=True)
        verifier = RecoveryVerifier(cooldown_sec=30)
        result = verify_after_remediation(report, _frame([UNHEALTHY]), _frame([HEALTHY]), "payment-api", verifier=verifier)
        self.assertIsNone(result)

    def test_runs_when_actions_executed(self):
        executor = RemediationExecutor(MockBackend(), mode="approval", audit_path=None)
        plan = [ActionSpec(name="clear_cache", description="", params={"service": "payment-api"})]
        report = executor.execute(plan, approve=True)
        self.assertEqual(report.executed, 1)
        verifier = RecoveryVerifier(cooldown_sec=30)
        result = verify_after_remediation(report, _frame([UNHEALTHY]), _frame([HEALTHY]), "payment-api", action="clear_cache", verifier=verifier)
        self.assertIsInstance(result, RecoveryReport)
        self.assertEqual(result.status, "SUCCESS")


if __name__ == "__main__":
    unittest.main()