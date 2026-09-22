import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.risk import RiskEngine, level_from_score
from src.risk.engine import _percentile_scores


class _StubModel:
    def __init__(self, probs, features=None):
        self._probs = np.asarray(probs, dtype=float)
        self.feature_columns_in = features

    def predict_proba(self, X):
        return np.column_stack([1.0 - self._probs, self._probs])


def _frame(n_rows=10, n_services=2):
    rows = []
    for svc in range(n_services):
        for i in range(n_rows):
            rows.append(
                {
                    "timestamp": pd.Timestamp("2026-09-19 00:00:00+00:00") + pd.Timedelta(minutes=1) * i,
                    "service": f"svc-{svc}",
                    "Error_rate": 0.0,
                    "Latency_15min_avg": 10.0,
                    "Memory_growth_rate": 0.0,
                    "feature_a": 1.0,
                }
            )
    df = pd.DataFrame(rows)
    df["Error_rate"] = np.linspace(0.0, 0.9, len(df))
    df["Latency_15min_avg"] = np.linspace(5.0, 95.0, len(df))
    return df


class TestLevelFromScore(unittest.TestCase):
    THRESHOLDS = {"WARNING": 0.40, "ANOMALOUS": 0.65, "CRITICAL": 0.85}

    def test_level_boundaries(self):
        self.assertEqual(level_from_score(0.1, self.THRESHOLDS), "NORMAL")
        self.assertEqual(level_from_score(0.3999, self.THRESHOLDS), "NORMAL")
        self.assertEqual(level_from_score(0.40, self.THRESHOLDS), "WARNING")
        self.assertEqual(level_from_score(0.6499, self.THRESHOLDS), "WARNING")
        self.assertEqual(level_from_score(0.65, self.THRESHOLDS), "ANOMALOUS")
        self.assertEqual(level_from_score(0.85, self.THRESHOLDS), "CRITICAL")
        self.assertEqual(level_from_score(1.0, self.THRESHOLDS), "CRITICAL")


class TestPercentile(unittest.TestCase):
    def test_bounds_and_monotonicity(self):
        ref = np.sort(np.arange(0.0, 100.0, 1.0))
        values = np.array([-5.0, 0.0, 49.0, 99.0, 150.0])
        scores = _percentile_scores(values, ref)
        self.assertTrue(np.all((scores >= 0.0) & (scores <= 1.0)))
        self.assertEqual(scores[0], 0.0)
        self.assertEqual(scores[-1], 1.0)
        self.assertTrue(np.all(np.diff(scores) >= 0))

    def test_empty_reference(self):
        scores = _percentile_scores(np.array([0.5]), np.array([]))
        self.assertTrue(np.all(np.isnan(scores)))


class TestRiskEngine(unittest.TestCase):
    def setUp(self):
        self.weights = {
            "failure_probability": 0.4,
            "error_rate": 0.15,
            "latency_trend": 0.1,
            "memory_growth": 0.1,
            "anomaly_score": 0.25,
        }
        self.thresholds = {"WARNING": 0.4, "ANOMALOUS": 0.65, "CRITICAL": 0.85}
        self.signal_map = {
            "error_rate": "Error_rate",
            "latency_trend": "Latency_15min_avg",
            "memory_growth": "Memory_growth_rate",
        }

    def _engine(self, probs=None):
        train = _frame(n_rows=20, n_services=2)
        probs = np.asarray(probs if probs is not None else np.linspace(0.3, 0.8, len(train)))
        engine = RiskEngine(
            models={"logistic_regression": _StubModel(probs, ["feature_a"])},
            feature_columns=["feature_a"],
            weights=self.weights,
            severity_thresholds=self.thresholds,
            signal_map=self.signal_map,
        )
        engine.calibrate(train)
        return engine, train

    def test_component_weights_renormalized_without_anomaly(self):
        engine, _ = self._engine()
        weights = engine._component_weights()
        self.assertNotIn("anomaly_score", weights)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)

    def test_score_output_shape_and_range(self):
        engine, df = self._engine()
        scored = engine.score(df)
        for col in ("timestamp", "service", "failure_probability", "error_rate", "latency_trend", "memory_growth", "risk_score", "risk_level", "risk"):
            self.assertIn(col, scored.columns)
        self.assertEqual(len(scored), len(df))
        self.assertTrue(np.all((scored["risk_score"] >= 0.0) & (scored["risk_score"] <= 1.0)))
        self.assertTrue(np.all((scored["failure_probability"] >= 0.0) & (scored["failure_probability"] <= 1.0)))
        self.assertTrue(set(scored["risk_level"]).issubset({"NORMAL", "WARNING", "ANOMALOUS", "CRITICAL"}))
        self.assertTrue(set(scored["risk"].unique()).issubset({0, 1}))

    def test_score_raises_without_timestamp_service(self):
        engine, _ = self._engine()
        with self.assertRaises(ValueError):
            engine.score(pd.DataFrame({"feature_a": [1.0]}))

    def test_latest_by_service_takes_newest_per_service(self):
        engine, df = self._engine()
        latest = engine.latest_by_service(df)
        self.assertEqual(len(latest), df["service"].nunique())
        for _, row in latest.iterrows():
            newest = df[df["service"] == row["service"]]["timestamp"].max()
            self.assertEqual(row["timestamp"], newest)

    def test_load_from_disk(self):
        backend = Path(__file__).resolve().parents[1]
        models_dir = backend / "models"
        if not (models_dir / "baseline" / "logistic_regression").exists():
            self.skipTest("trained models not present")
        engine = RiskEngine.load(models_dir, {})
        self.assertIn("failure_probability", engine.available_components())
        df = _frame(n_rows=14, n_services=3)
        df["feature_a"] = 1.0
        for feature in engine.feature_columns:
            df[feature] = 0.0
        scored = engine.score(df)
        self.assertIn("risk_score", scored.columns)
        self.assertTrue(np.all((scored["risk_score"] >= 0.0) & (scored["risk_score"] <= 1.0)))


if __name__ == "__main__":
    unittest.main()