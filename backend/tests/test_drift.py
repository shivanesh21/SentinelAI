from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.mlops import (
    DataDriftSummary,
    DriftDetector,
    DriftMonitoringEngine,
    FeatureDriftResult,
    PerformanceMetrics,
    PerformanceMonitor,
    PerformanceReport,
    RetrainingTrigger,
    calculate_ks_test,
    calculate_psi,
)


class TestPSI(unittest.TestCase):
    def test_identical_distributions(self):
        np.random.seed(42)
        ref = np.random.normal(50, 10, 1000)
        target = ref.copy()
        psi = calculate_psi(ref, target, num_bins=10)
        self.assertLess(psi, 0.01)

    def test_similar_distributions(self):
        np.random.seed(42)
        ref = np.random.normal(50, 10, 2000)
        target = np.random.normal(50, 10, 2000)
        psi = calculate_psi(ref, target, num_bins=10)
        self.assertLess(psi, 0.10)

    def test_severely_shifted_distribution(self):
        np.random.seed(42)
        ref = np.random.normal(50, 10, 2000)
        target = np.random.normal(90, 15, 2000)
        psi = calculate_psi(ref, target, num_bins=10)
        self.assertGreater(psi, 0.25)

    def test_empty_arrays(self):
        self.assertEqual(calculate_psi([], []), 0.0)
        self.assertEqual(calculate_psi([1.0, 2.0], []), 0.0)

    def test_nan_handling(self):
        ref = [1.0, 2.0, np.nan, 4.0, 5.0] * 50
        target = [1.0, 2.0, 3.0, 4.0, np.nan] * 50
        psi = calculate_psi(ref, target)
        self.assertIsInstance(psi, float)
        self.assertFalse(np.isnan(psi))

    def test_constant_arrays(self):
        ref = [5.0] * 100
        target = [5.0] * 100
        self.assertEqual(calculate_psi(ref, target), 0.0)

        target_diff = [10.0] * 100
        self.assertEqual(calculate_psi(ref, target_diff), 1.0)


class TestKSTest(unittest.TestCase):
    def test_identical_distributions(self):
        np.random.seed(42)
        ref = np.random.normal(0, 1, 500)
        target = ref.copy()
        stat, pval = calculate_ks_test(ref, target)
        self.assertEqual(stat, 0.0)
        self.assertEqual(pval, 1.0)

    def test_significantly_different_distributions(self):
        np.random.seed(42)
        ref = np.random.normal(0, 1, 500)
        target = np.random.normal(5, 1, 500)
        stat, pval = calculate_ks_test(ref, target)
        self.assertGreater(stat, 0.8)
        self.assertLess(pval, 1e-5)

    def test_empty_inputs(self):
        stat, pval = calculate_ks_test([], [])
        self.assertEqual(stat, 0.0)
        self.assertEqual(pval, 1.0)

    def test_constant_distributions(self):
        stat, pval = calculate_ks_test([2.0] * 20, [2.0] * 20)
        self.assertEqual(stat, 0.0)
        self.assertEqual(pval, 1.0)

        stat, pval = calculate_ks_test([2.0] * 20, [5.0] * 20)
        self.assertEqual(stat, 1.0)
        self.assertEqual(pval, 0.0)


class TestDriftDetector(unittest.TestCase):
    def setUp(self):
        self.detector = DriftDetector(
            num_bins=10,
            moderate_psi=0.10,
            significant_psi=0.25,
            ks_alpha=0.05,
            dataset_drift_share_threshold=0.30,
        )

    def test_analyze_feature_nominal(self):
        np.random.seed(42)
        ref = np.random.normal(10, 2, 500)
        target = np.random.normal(10, 2, 500)
        res = self.detector.analyze_feature(ref, target, feature_name="latency")
        self.assertIsInstance(res, FeatureDriftResult)
        self.assertEqual(res.drift_level, "none")
        self.assertFalse(res.drift_detected)

    def test_analyze_feature_significant_drift(self):
        np.random.seed(42)
        ref = np.random.normal(10, 2, 500)
        target = np.random.normal(25, 5, 500)
        res = self.detector.analyze_feature(ref, target, feature_name="latency")
        self.assertEqual(res.drift_level, "significant")
        self.assertTrue(res.drift_detected)
        self.assertGreater(res.psi, 0.25)

    def test_analyze_dataset(self):
        np.random.seed(42)
        n = 500
        ref_df = pd.DataFrame({
            "f1": np.random.normal(10, 2, n),
            "f2": np.random.normal(50, 5, n),
            "f3": np.random.normal(100, 10, n),
            "f4": np.random.normal(0, 1, n),
        })
        # Drift 2 out of 4 features (50% >= 30%)
        target_df = pd.DataFrame({
            "f1": np.random.normal(10, 2, n),
            "f2": np.random.normal(50, 5, n),
            "f3": np.random.normal(180, 20, n),  # drifted
            "f4": np.random.normal(10, 2, n),    # drifted
        })

        summary, results = self.detector.analyze(ref_df, target_df)
        self.assertIsInstance(summary, DataDriftSummary)
        self.assertEqual(summary.total_features, 4)
        self.assertEqual(summary.drifted_features_count, 2)
        self.assertEqual(summary.drift_share, 0.5)
        self.assertTrue(summary.dataset_drift)
        self.assertIn("f3", summary.drifted_features)
        self.assertIn("f4", summary.drifted_features)


class TestPerformanceMonitor(unittest.TestCase):
    def setUp(self):
        self.monitor = PerformanceMonitor(
            f1_drop_threshold=0.15,
            min_f1_threshold=0.60,
            min_samples=10,
        )

    def test_compute_metrics(self):
        y_true = [1, 1, 0, 0, 1, 0, 1, 0]
        y_pred = [1, 1, 0, 0, 0, 0, 1, 0]
        m = self.monitor.compute_metrics(y_true, y_pred)
        self.assertEqual(m.sample_count, 8)
        self.assertEqual(m.precision, 1.0)
        self.assertEqual(m.recall, 0.75)
        self.assertAlmostEqual(m.f1, 0.8571, places=3)

    def test_evaluate_nominal(self):
        y_true = [1] * 30 + [0] * 30
        y_pred = [1] * 28 + [0] * 2 + [0] * 30
        rep = self.monitor.evaluate(y_true, y_pred, baseline_f1=0.90)
        self.assertFalse(rep.degraded)
        self.assertEqual(rep.status, "ok")
        self.assertLess(rep.f1_drop, 0.15)

    def test_evaluate_degraded(self):
        y_true = [1] * 30 + [0] * 30
        y_pred = [1] * 5 + [0] * 25 + [0] * 30  # severe recall drop
        rep = self.monitor.evaluate(y_true, y_pred, baseline_f1=0.90)
        self.assertTrue(rep.degraded)
        self.assertEqual(rep.status, "degraded")
        self.assertGreaterEqual(rep.f1_drop, 0.15)

    def test_insufficient_samples(self):
        rep = self.monitor.evaluate([1, 0, 1], [1, 0, 0], baseline_f1=0.90)
        self.assertEqual(rep.status, "insufficient_data")
        self.assertFalse(rep.degraded)


class TestRetrainingTrigger(unittest.TestCase):
    def setUp(self):
        self.trigger = RetrainingTrigger(
            drift_share_threshold=0.30,
            critical_features=["Error_rate", "Latency_15min_avg"],
        )

    def test_healthy_system(self):
        summary = DataDriftSummary(
            total_features=10,
            drifted_features_count=1,
            drift_share=0.10,
            dataset_drift=False,
            drifted_features=["minor_metric"],
        )
        perf = PerformanceReport(
            baseline_f1=0.85,
            rolling_metrics=PerformanceMetrics(100, 0.84, 0.85, 0.83, 0.90),
            f1_drop=0.01,
            degraded=False,
            status="ok",
        )
        res = self.trigger.evaluate(summary, {}, perf)
        self.assertFalse(res.retraining_required)
        self.assertEqual(res.severity, "normal")

    def test_dataset_drift_trigger(self):
        summary = DataDriftSummary(
            total_features=10,
            drifted_features_count=4,
            drift_share=0.40,
            dataset_drift=True,
            drifted_features=["f1", "f2", "f3", "f4"],
        )
        perf = PerformanceReport(
            baseline_f1=0.85,
            rolling_metrics=PerformanceMetrics(100, 0.83, 0.85, 0.81, 0.90),
            f1_drop=0.02,
            degraded=False,
            status="ok",
        )
        res = self.trigger.evaluate(summary, {}, perf)
        self.assertTrue(res.retraining_required)
        self.assertEqual(res.severity, "warning")
        self.assertTrue(any("Dataset drift threshold breached" in r for r in res.trigger_reasons))

    def test_critical_feature_drift_trigger(self):
        summary = DataDriftSummary(
            total_features=10,
            drifted_features_count=1,
            drift_share=0.10,
            dataset_drift=False,
            drifted_features=["Error_rate"],
        )
        crit_feat = FeatureDriftResult(
            feature="Error_rate",
            psi=0.45,
            ks_statistic=0.35,
            ks_pvalue=0.0001,
            drift_level="significant",
            drift_detected=True,
            reference_mean=0.01,
            target_mean=0.08,
            mean_shift=0.07,
            reference_std=0.01,
            target_std=0.03,
        )
        perf = PerformanceReport(
            baseline_f1=0.85,
            rolling_metrics=PerformanceMetrics(100, 0.84, 0.85, 0.83, 0.90),
            f1_drop=0.01,
            degraded=False,
            status="ok",
        )
        res = self.trigger.evaluate(summary, {"Error_rate": crit_feat}, perf)
        self.assertTrue(res.retraining_required)
        self.assertTrue(any("Critical operational feature" in r for r in res.trigger_reasons))

    def test_performance_degradation_trigger(self):
        summary = DataDriftSummary(
            total_features=10,
            drifted_features_count=0,
            drift_share=0.0,
            dataset_drift=False,
        )
        perf = PerformanceReport(
            baseline_f1=0.88,
            rolling_metrics=PerformanceMetrics(100, 0.55, 0.70, 0.45, 0.80),
            f1_drop=0.33,
            degraded=True,
            status="degraded",
        )
        res = self.trigger.evaluate(summary, {}, perf)
        self.assertTrue(res.retraining_required)
        self.assertEqual(res.severity, "critical")
        self.assertTrue(any("performance degradation" in r for r in res.trigger_reasons))


class TestDriftMonitoringEngine(unittest.TestCase):
    def test_engine_run_and_save(self):
        np.random.seed(42)
        n = 100
        ref_df = pd.DataFrame({
            "Error_rate": np.random.beta(1, 20, n),
            "Latency_15min_avg": np.random.normal(120, 15, n),
            "failure_in_next_10min": np.random.binomial(1, 0.1, n),
        })
        target_df = pd.DataFrame({
            "Error_rate": np.random.beta(5, 10, n),  # severe shift
            "Latency_15min_avg": np.random.normal(350, 40, n),  # severe shift
            "failure_in_next_10min": np.random.binomial(1, 0.1, n),
        })

        engine = DriftMonitoringEngine()
        report = engine.run(
            reference_df=ref_df,
            target_df=target_df,
            y_true=target_df["failure_in_next_10min"].values,
            y_pred=target_df["failure_in_next_10min"].values,
            baseline_f1=0.85,
        )

        self.assertTrue(report.data_drift.dataset_drift)
        self.assertTrue(report.trigger.retraining_required)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "drift_report.json"
            saved = engine.save_report(report, out_file)
            self.assertTrue(saved.exists())
            self.assertGreater(saved.stat().st_size, 100)


if __name__ == "__main__":
    unittest.main()
