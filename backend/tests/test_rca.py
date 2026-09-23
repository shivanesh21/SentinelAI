import os
import unittest
from pathlib import Path

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


class TestParsing(unittest.TestCase):
    def test_extract_json_strips_fences(self):
        from src.rca import extract_json, parse_result

        raw = '```json\n{"probable_root_cause": "a", "confidence_score": 0.9, "recommended_action": "b"}\n```'
        self.assertEqual(extract_json(raw)["confidence_score"], 0.9)
        self.assertEqual(parse_result(raw).probable_root_cause, "a")

    def test_extract_json_raises_without_object(self):
        from src.rca import extract_json

        with self.assertRaises(ValueError):
            extract_json("no JSON here")

    def test_parse_result_rejects_missing_keys(self):
        from src.rca import parse_result

        with self.assertRaises(ValueError):
            parse_result('{"confidence_score": 0.5}')


class TestAuditLogger(unittest.TestCase):
    def test_record_and_read_round_trip(self):
        from tempfile import TemporaryDirectory

        from src.rca import AuditLogger, RCAEvent

        with TemporaryDirectory() as tmp:
            logger = AuditLogger(tmp + "/audit.jsonl")
            event = RCAEvent(
                request_id="abc123",
                timestamp="2026-09-19T04:30:08+00:00",
                provider="heuristic",
                service="payment-api",
                evidence={"service": "payment-api", "risk_score": 0.9},
                result={"root_cause_category": "memory_leak"},
                status="ok",
                duration_ms=1.5,
            )
            logger.record(event)
            records = logger.read()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["request_id"], "abc123")
            self.assertEqual(records[0]["result"]["root_cause_category"], "memory_leak")
            self.assertEqual(records[0]["provider"], "heuristic")

    def test_read_limits(self):
        from tempfile import TemporaryDirectory

        from src.rca import AuditLogger, RCAEvent

        with TemporaryDirectory() as tmp:
            logger = AuditLogger(tmp + "/audit.jsonl")
            for i in range(5):
                logger.record(
                    RCAEvent(
                        request_id=f"r{i}",
                        timestamp="t",
                        provider="heuristic",
                        service="s",
                        evidence={},
                        result=None,
                        status="ok",
                        duration_ms=0.0,
                    )
                )
            self.assertEqual(len(logger.read(20)), 5)
            self.assertEqual([r["request_id"] for r in logger.read(2)], ["r3", "r4"])
            self.assertEqual(logger.read(0), [])


class TestMalformedOutputFallback(unittest.TestCase):
    def _broken_client(self, raw, audit=None):
        client = RCAClient(provider="openai", audit=audit)
        client._call_openai = lambda evidence: raw
        return client

    def test_garbage_payload_falls_back_to_heuristic(self):
        from tempfile import TemporaryDirectory

        from src.rca import AuditLogger

        with TemporaryDirectory() as tmp:
            logger = AuditLogger(tmp + "/audit.jsonl")
            client = self._broken_client("this is not json at all", audit=logger)
            result = client.analyze(evidence_for("network_partition"))
            self.assertEqual(result.root_cause_category, "network_partition")
            records = logger.read()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["status"], "fallback_heuristic")
            self.assertIn("error", records[0])
            self.assertEqual(records[0]["result"]["root_cause_category"], "network_partition")

    def test_valid_but_schema_invalid_falls_back(self):
        from tempfile import TemporaryDirectory

        from src.rca import AuditLogger

        with TemporaryDirectory() as tmp:
            logger = AuditLogger(tmp + "/audit.jsonl")
            client = self._broken_client('{"probable_root_cause": ""}', audit=logger)
            result = client.analyze(evidence_for("disk_fill"))
            self.assertEqual(result.root_cause_category, "disk_fill")
            self.assertEqual(logger.read(1)[0]["status"], "fallback_heuristic")


class TestEndToEnd(unittest.TestCase):
    def test_risk_engine_to_rca_output(self):
        from tempfile import TemporaryDirectory

        import pandas as pd

        from src.evidence import EvidenceAssembler
        from src.risk import RiskEngine

        backend = Path(__file__).resolve().parents[1]
        models_dir = backend / "models"
        test_parquet = backend / "data" / "splits" / "test.parquet"
        if not (models_dir / "baseline" / "logistic_regression").exists() or not test_parquet.exists():
            self.skipTest("trained models or test split not present")

        engine = RiskEngine.load(models_dir, {})
        test = pd.read_parquet(test_parquet)
        slice_df = (
            test[test["service"] == "payment-api"]
            .sort_values("timestamp")
            .tail(40)
            .drop(columns=["in_failure", "time_to_failure_min"], errors="ignore")
        )

        assembler = EvidenceAssembler(engine, threshold=0.40, feature_columns=engine.feature_columns)
        packages = assembler.assemble(slice_df)
        if not packages:
            self.skipTest("no risk crossing threshold in slice")

        from src.rca import AuditLogger

        with TemporaryDirectory() as tmp:
            audit = AuditLogger(tmp + "/audit.jsonl")
            client = RCAClient(audit=audit)
            finding = client.analyze(packages[0].to_dict())
            self.assertTrue(audit.path.exists())

        self.assertIsInstance(finding, RCAResult)
        self.assertIn(finding.root_cause_category, CAUSE_CATEGORIES)
        self.assertTrue(finding.probable_root_cause)
        self.assertTrue(0.0 <= finding.confidence_score <= 1.0)
        self.assertTrue(finding.evidence)


if __name__ == "__main__":
    unittest.main()