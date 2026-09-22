import os
import unittest

from src.rca import (
    CAUSE_CATEGORIES,
    RCAClient,
    build_rca_prompt,
    heuristic_classify,
    resolve_provider,
)
from src.rca.schema import RCAResult


def evidence_for(category: str) -> dict:
    """Synthetic evidence vector matching a specific incident type."""
    base = {
        "failure_probability": 0.7,
        "components": {
            "failure_probability": 0.7,
            "error_rate": 0.4,
            "latency_trend": 0.4,
            "memory_growth": 0.2,
        },
        "top_anomalous_features": [],
        "metrics_deltas": {},
        "correlated_logs": [],
    }
    if category == "memory_leak":
        base["components"]["memory_growth"] = 0.95
        base["top_anomalous_features"].append(
            {"feature": "Memory_growth_rate", "zscore": 6.0, "direction": "up"}
        )
        base["correlated_logs"] = [
            {"level": "WARN", "message": "heap usage > 90% - gc pauses"},
        ]
    elif category == "connection_pool_exhaustion":
        base["top_anomalous_features"].append(
            {"feature": "Request_rate", "zscore": 4.0, "direction": "up"}
        )
        base["correlated_logs"] = [
            {"level": "ERROR", "message": "connection pool exhausted - 0 available after acquire timeout"},
        ]
    elif category == "latency_spike":
        base["top_anomalous_features"].append(
            {"feature": "Latency_5min_avg", "zscore": 5.0, "direction": "up"}
        )
        base["correlated_logs"] = [
            {"level": "WARN", "message": "p99 latency 1.1s exceeded SLO"},
        ]
    elif category == "network_partition":
        base["top_anomalous_features"].append(
            {"feature": "Network_in_5min_avg", "zscore": -6.0, "direction": "down"}
        )
        base["correlated_logs"] = [
            {"level": "ERROR", "message": "broker unreachable - connection refused"},
        ]
    elif category == "disk_fill":
        base["top_anomalous_features"].append(
            {"feature": "Disk_growth_rate", "zscore": 5.0, "direction": "up"}
        )
        base["correlated_logs"] = [
            {"level": "ERROR", "message": "no space left on device"},
        ]
    return base


class TestPrompt(unittest.TestCase):
    def test_prompt_includes_schema_and_evidence(self):
        system, user = build_rca_prompt(evidence_for("memory_leak"))
        for key in (
            "probable_root_cause",
            "root_cause_category",
            "evidence",
            "confidence_score",
            "recommended_action",
        ):
            self.assertIn(key, system)
        self.assertIn("failure_probability", user)

    def test_prompt_accepts_json_string(self):
        system, user = build_rca_prompt('{"risk_score": 0.9}')
        self.assertIn("0.9", user)

    def test_categories_are_stable(self):
        self.assertIn("memory_leak", CAUSE_CATEGORIES)
        self.assertIn("connection_pool_exhaustion", CAUSE_CATEGORIES)
        self.assertIn("unknown", CAUSE_CATEGORIES)


class TestSchema(unittest.TestCase):
    def test_round_trip(self):
        result = RCAResult(
            probable_root_cause="leak",
            confidence_score=0.9,
            recommended_action="heal",
            root_cause_category="memory_leak",
            evidence=["a", "b"],
        )
        restored = RCAResult.from_dict(result.to_dict())
        self.assertEqual(restored, result)

    def test_rejects_bad_confidence(self):
        with self.assertRaises(ValueError):
            RCAResult.from_dict(
                {"probable_root_cause": "x", "confidence_score": 1.5, "recommended_action": "y"}
            )

    def test_rejects_missing_keys(self):
        with self.assertRaises(ValueError):
            RCAResult.from_dict({"probable_root_cause": "x"})


class TestHeuristic(unittest.TestCase):
    def test_classifies_all_types(self):
        expected = {
            "memory_leak",
            "connection_pool_exhaustion",
            "latency_spike",
            "network_partition",
            "disk_fill",
        }
        for category in expected:
            with self.subTest(category=category):
                result = heuristic_classify(evidence_for(category))
                self.assertEqual(result.root_cause_category, category)
                self.assertGreaterEqual(result.confidence_score, 0.0)
                self.assertLessEqual(result.confidence_score, 1.0)
                self.assertTrue(result.probable_root_cause)
                self.assertTrue(result.recommended_action)
                self.assertTrue(result.evidence)

    def test_unknown_on_weak_signals(self):
        result = heuristic_classify({"components": {}, "top_anomalous_features": [], "correlated_logs": []})
        self.assertEqual(result.root_cause_category, "unknown")


class TestClient(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("LLM_PROVIDER", None)
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("LOCAL_LLM_URL", None)
        os.environ.pop("ANTHROPIC_MODEL", None)

    def test_resolve_provider_falls_back_to_heuristic(self):
        os.environ.pop("LLM_PROVIDER", None)
        self.assertEqual(resolve_provider(), "heuristic")

    def test_resolve_anthropic_requires_key(self):
        os.environ["LLM_PROVIDER"] = "anthropic"
        os.environ.pop("ANTHROPIC_API_KEY", None)
        self.assertEqual(resolve_provider(), "heuristic")
        os.environ["ANTHROPIC_API_KEY"] = "sk-test"
        self.assertEqual(resolve_provider(), "anthropic")

    def test_client_analyzes_with_heuristic_when_unconfigured(self):
        os.environ.pop("LLM_PROVIDER", None)
        client = RCAClient()
        self.assertEqual(client.provider, "heuristic")
        result = client.analyze(evidence_for("memory_leak"))
        self.assertIsInstance(result, RCAResult)
        self.assertEqual(result.root_cause_category, "memory_leak")


if __name__ == "__main__":
    unittest.main()