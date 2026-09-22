from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.features.engineering import (
    DERIVED_FEATURES,
    FeatureConfig,
    FeatureEngineer,
    build_features,
    feature_config_from_settings,
)
from src.telemetry.simulator.schema import METRIC_FIELDS

START = datetime(2026, 9, 18, tzinfo=timezone.utc)


def make_metrics(services: list[tuple[str, dict[str, float]]], ticks: int = 60, interval_sec: int = 30) -> pd.DataFrame:
    rows = []
    for service, values in services:
        for i in range(ticks):
            row = {
                "timestamp": START + timedelta(seconds=i * interval_sec),
                "service": service,
                "healthy": True,
            }
            for field in METRIC_FIELDS:
                row[field] = values.get(field, 50.0)
            rows.append(row)
    return pd.DataFrame(rows)


class RollingFeatureTests(unittest.TestCase):
    def test_required_feature_names_present(self) -> None:
        engineer = FeatureEngineer(FeatureConfig(windows_min=[5, 10, 15]))
        required = {
            "CPU_5min_avg",
            "Memory_10min_avg",
            "Latency_5min_avg",
            "CPU_growth_rate",
            "Memory_growth_rate",
            "Error_rate",
            "Request_rate",
            "CPU_memory_ratio",
            "latency_per_request",
        }
        self.assertTrue(required.issubset(set(engineer.feature_names)))
        self.assertTrue(set(DERIVED_FEATURES).issubset(set(engineer.feature_names)))

    def test_constant_series_rolling_mean(self) -> None:
        metrics = make_metrics([("svc-a", {"cpu_util_pct": 42.0})])
        features = build_features(metrics)
        self.assertTrue((features["CPU_5min_avg"] == 42.0).all())
        self.assertTrue((features["CPU_15min_avg"] == 42.0).all())

    def test_rolling_window_value(self) -> None:
        metrics = make_metrics([("svc-a", {})], ticks=30)
        metrics["cpu_util_pct"] = [float(i) for i in range(30)]
        features = build_features(metrics)
        config = FeatureConfig()
        ticks = config.window_ticks(5)
        expected_last = sum(range(30 - ticks, 30)) / ticks
        self.assertAlmostEqual(features["CPU_5min_avg"].iloc[-1], expected_last)
        self.assertEqual(features["CPU_5min_avg"].iloc[0], 0.0)

    def test_no_nan_after_warmup(self) -> None:
        metrics = make_metrics([("svc-a", {}), ("svc-b", {})], ticks=80)
        features = build_features(metrics)
        feature_cols = [c for c in features.columns if c not in ("timestamp", "service", "healthy")]
        self.assertEqual(int(features[feature_cols].isna().sum().sum()), 0)


class GrowthFeatureTests(unittest.TestCase):
    def test_linear_growth_rate(self) -> None:
        metrics = make_metrics([("svc-a", {})], ticks=60)
        metrics["cpu_util_pct"] = [100.0 + i for i in range(60)]
        features = build_features(metrics)
        ticks = FeatureConfig().growth_ticks
        current, previous = 100.0 + 40, 100.0 + (40 - ticks)
        self.assertAlmostEqual(features["CPU_growth_rate"].iloc[40], (current - previous) / previous)

    def test_growth_zero_warmup(self) -> None:
        metrics = make_metrics([("svc-a", {})], ticks=60)
        metrics["memory_util_pct"] = [200.0 + i for i in range(60)]
        features = build_features(metrics)
        self.assertEqual(features["Memory_growth_rate"].iloc[0], 0.0)


class CrossMetricFeatureTests(unittest.TestCase):
    def test_cpu_memory_ratio(self) -> None:
        metrics = make_metrics([("svc-a", {"cpu_util_pct": 40.0, "memory_util_pct": 80.0})])
        features = build_features(metrics)
        self.assertTrue((features["CPU_memory_ratio"] == 0.5).all())

    def test_latency_per_request_floor(self) -> None:
        metrics = make_metrics([("svc-a", {"latency_ms": 300.0, "request_rate_rps": 0.0})])
        features = build_features(metrics)
        self.assertTrue((features["latency_per_request"] == 300.0).all())

    def test_error_rate_is_rolling_sum(self) -> None:
        metrics = make_metrics([("svc-a", {})], ticks=40)
        metrics["http_4xx_rate"] = 0.01
        metrics["http_5xx_rate"] = 0.02
        features = build_features(metrics)
        self.assertAlmostEqual(features["Error_rate"].iloc[-1], 0.03)

    def test_request_rate_rolling(self) -> None:
        metrics = make_metrics([("svc-a", {})], ticks=40)
        metrics["request_rate_rps"] = 100.0
        features = build_features(metrics)
        self.assertTrue((features["Request_rate"] == 100.0).all())


class CausalAndIsolationTests(unittest.TestCase):
    def test_service_isolation(self) -> None:
        metrics = make_metrics([("svc-a", {"cpu_util_pct": 10.0}), ("svc-b", {"cpu_util_pct": 90.0})], ticks=40)
        features = build_features(metrics)
        a = features[features["service"] == "svc-a"]["CPU_5min_avg"]
        b = features[features["service"] == "svc-b"]["CPU_5min_avg"]
        self.assertTrue((a == 10.0).all())
        self.assertTrue((b == 90.0).all())

    def test_no_lookahead_leakage(self) -> None:
        metrics = make_metrics([("svc-a", {})], ticks=60)
        metrics["cpu_util_pct"] = [float(i) for i in range(60)]
        baseline = build_features(metrics)

        perturbed = metrics.copy()
        perturbed.loc[59, "cpu_util_pct"] = 9999.0
        changed = build_features(perturbed)

        feature_cols = [c for c in baseline.columns if c not in ("timestamp", "service", "healthy")]
        tick = 59 - FeatureConfig().window_ticks(15)
        pd.testing.assert_frame_equal(
            baseline.loc[: tick, feature_cols].reset_index(drop=True),
            changed.loc[: tick, feature_cols].reset_index(drop=True),
        )

    def test_input_validation(self) -> None:
        metrics = make_metrics([("svc-a", {})], ticks=10).drop(columns=["latency_ms"])
        with self.assertRaises(ValueError):
            build_features(metrics)


class ConfigTests(unittest.TestCase):
    def test_config_from_settings(self) -> None:
        settings = {
            "telemetry": {"collection_interval_sec": 30, "feature_windows_min": [5]},
            "features": {"windows_min": [5, 10], "growth_window_min": 15},
        }
        config = feature_config_from_settings(settings)
        self.assertEqual(config.windows_min, [5, 10])
        self.assertEqual(config.growth_window_min, 15)
        self.assertEqual(config.window_ticks(5), 10)
        self.assertEqual(config.window_ticks(10), 20)


if __name__ == "__main__":
    unittest.main()
