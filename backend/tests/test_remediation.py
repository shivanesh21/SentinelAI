import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.remediation import (
    ACTION_NAMES,
    ApprovalStore,
    MockBackend,
    RemediationExecutor,
    RemediationPlanner,
    RemediationPolicy,
    execute_spec,
    plan_for_category,
    policy_from_config,
    resolve_mode,
)
from src.remediation.model import ActionSpec, ExecutionReport


class TestPlanner(unittest.TestCase):
    def test_every_category_maps_to_actions(self):
        categories = [
            "memory_leak",
            "connection_pool_exhaustion",
            "latency_spike",
            "network_partition",
            "disk_fill",
            "code_deployment",
            "traffic_spike",
        ]
        for category in categories:
            with self.subTest(category=category):
                spec = ActionSpec(name="x", description="", params={"service": "s"})
                plan = plan_for_category(category, "svc-a")
                self.assertGreater(len(plan), 0)
                for action in plan:
                    self.assertIn(action.name, ACTION_NAMES)
                    self.assertIn(action.risk, ("low", "medium", "high"))

    def test_unknown_category_returns_ticket_only(self):
        plan = plan_for_category("bogus", "svc-a")
        self.assertEqual([a.name for a in plan], ["create_incident_ticket"])

    def test_scale_delta_applied_to_current_replicas(self):
        plan = plan_for_category("traffic_spike", "web", context={"current_replicas": 2})
        scale = [a for a in plan if a.name == "scale_replicas"][0]
        self.assertEqual(scale.params["count"], 4)

    def test_restart_spec_has_container_param(self):
        plan = plan_for_category("network_partition", "worker-1")
        restart = [a for a in plan if a.name == "restart_container"][0]
        self.assertEqual(restart.params["container"], "worker-1")


class TestExecutor(unittest.TestCase):
    def setUp(self):
        self.plan = plan_for_category("traffic_spike", "web-gateway", context={"current_replicas": 1})

    def test_approval_mode_dry_runs_by_default(self):
        executor = RemediationExecutor(backend=MockBackend(), mode="approval", audit_path=None)
        report = executor.execute(self.plan)
        self.assertTrue(report.dry_run)
        self.assertEqual(report.executed, 0)
        self.assertTrue(all(r.status == "planned" for r in report.results))

    def test_advisory_mode_never_executes(self):
        executor = RemediationExecutor(backend=MockBackend(), mode="advisory", audit_path=None)
        report = executor.execute(self.plan, approve=True)
        self.assertTrue(report.dry_run)
        self.assertEqual(report.executed, 0)

    def test_approved_execution_runs_against_backend(self):
        backend = MockBackend()
        executor = RemediationExecutor(backend=backend, mode="approval", audit_path=None)
        report = executor.execute(self.plan, approve=True)
        self.assertFalse(report.dry_run)
        self.assertEqual(report.executed, 3)
        self.assertTrue(report.all_ok)
        self.assertEqual(backend.replicas.get("web-gateway"), 3)
        self.assertEqual(len(backend.tickets), 1)

    def test_autonomous_mode_executes_whitelisted(self):
        backend = MockBackend()
        policy = RemediationPolicy(
            autonomous_whitelist=["scale_replicas", "clear_cache", "create_incident_ticket"]
        )
        executor = RemediationExecutor(backend=backend, mode="autonomous", audit_path=None, policy=policy)
        report = executor.execute(self.plan)
        self.assertFalse(report.dry_run)
        self.assertEqual(report.executed, 3)
        self.assertEqual(report.pending, 0)

    def test_autonomous_gates_non_whitelisted_actions(self):
        backend = MockBackend()
        executor = RemediationExecutor(backend=backend, mode="autonomous", audit_path=None)
        report = executor.execute(plan_for_category("latency_spike", "auth-service", {"current_replicas": 1}))
        statuses = {r.name: r.status for r in report.results}
        self.assertEqual(statuses["clear_cache"], "ok")
        self.assertEqual(statuses["scale_replicas"], "requires_approval")
        self.assertEqual(statuses["create_incident_ticket"], "requires_approval")
        self.assertEqual(report.executed, 1)
        self.assertEqual(report.pending, 2)

    def test_autonomous_respects_max_actions_per_incident(self):
        backend = MockBackend()
        plan = plan_for_category("latency_spike", "svc", {"current_replicas": 1})
        policy = RemediationPolicy(
            autonomous_whitelist=["scale_replicas", "clear_cache", "create_incident_ticket"],
            max_actions_per_incident=2,
        )
        executor = RemediationExecutor(backend=backend, mode="autonomous", audit_path=None, policy=policy)
        report = executor.execute(plan)
        self.assertEqual(report.executed, 2)
        self.assertEqual(report.pending, 1)

    def test_memory_leak_plan_executes_restart(self):
        plan = plan_for_category("memory_leak", "order-service")
        executor = RemediationExecutor(backend=MockBackend(), mode="approval", audit_path=None)
        report = executor.execute(plan, approve=True)
        self.assertTrue(report.all_ok)
        restart = next(r for r in report.results if r.name == "restart_container")
        self.assertEqual(restart.status, "ok")

    def test_rollback_restores_scale_state(self):
        backend = MockBackend()
        executor = RemediationExecutor(backend=backend, mode="approval", audit_path=None)
        report = executor.execute(self.plan, approve=True)
        rolled = executor.rollback(report)
        self.assertEqual(backend.replicas.get("web-gateway"), 1)
        statuses = {r.name: r.status for r in rolled.results}
        self.assertEqual(statuses["scale_replicas"], "rolled_back")
        self.assertEqual(statuses["create_incident_ticket"], "skipped")

    def test_audit_writes_jsonl(self):
        with TemporaryDirectory() as tmp:
            path = tmp + "/audit.jsonl"
            executor = RemediationExecutor(backend=MockBackend(), mode="approval", audit_path=path)
            executor.execute(self.plan, approve=True)
            records = self._read_jsonl(path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["mode"], "approval")
            self.assertEqual(len(records[0]["actions"]), 3)
            self.assertIn("event_id", records[0])

    @staticmethod
    def _read_jsonl(path):
        import json

        return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


class TestModeResolution(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("REMEDIATION_MODE", None)

    def test_invalid_mode_falls_back_to_approval(self):
        self.assertEqual(resolve_mode("chaotic"), "approval")

    def test_env_override(self):
        os.environ["REMEDIATION_MODE"] = "autonomous"
        self.assertEqual(resolve_mode(), "autonomous")
        self.assertEqual(resolve_mode("advisory"), "advisory")

    def test_default_is_approval(self):
        self.assertEqual(resolve_mode(), "approval")


class TestPolicy(unittest.TestCase):
    def test_default_whitelist_only_clear_cache(self):
        policy = RemediationPolicy()
        self.assertEqual(policy.autonomous_whitelist, {"clear_cache"})
        self.assertEqual(policy.max_risk_autonomous, "low")
        self.assertEqual(policy.max_actions_per_incident, 4)

    def test_allow_autonomous_checks_whitelist_and_risk(self):
        policy = RemediationPolicy()
        allowed, reason = policy.allow_autonomous(
            ActionSpec(name="clear_cache", description="", params={}, risk="low")
        )
        self.assertTrue(allowed)
        self.assertEqual(reason, "ok")
        denied, reason = policy.allow_autonomous(
            ActionSpec(name="restart_container", description="", params={}, risk="medium")
        )
        self.assertFalse(denied)
        self.assertIn("not in autonomous whitelist", reason)

    def test_medium_risk_whitelisted_action_still_blocked(self):
        policy = RemediationPolicy(autonomous_whitelist=["restart_container"])
        allowed, reason = policy.allow_autonomous(
            ActionSpec(name="restart_container", description="", params={}, risk="medium")
        )
        self.assertFalse(allowed)
        self.assertIn("risk", reason)

    def test_decisions_non_autonomous_are_all_proposals(self):
        plan = plan_for_category("network_partition", "svc")
        decisions = RemediationPolicy().decisions(plan, autonomous=False)
        self.assertTrue(all(d == "propose" for _, d, _ in decisions))

    def test_policy_from_config(self):
        config = {"remediation": {"policy": {"autonomous_whitelist": ["clear_cache", "scale_replicas"]}}}
        policy = policy_from_config(config)
        self.assertEqual(policy.autonomous_whitelist, {"clear_cache", "scale_replicas"})
        self.assertEqual(policy.to_dict()["max_actions_per_incident"], 4)


class TestApprovalStore(unittest.TestCase):
    def test_create_get_take_round_trip(self):
        store = ApprovalStore()
        plan = plan_for_category("disk_fill", "inventory")
        approval_id = store.create("req-1", "inventory", "disk_fill", plan)
        pending = store.get(approval_id)
        self.assertIsNotNone(pending)
        self.assertEqual(pending.request_id, "req-1")
        self.assertEqual(pending.category, "disk_fill")
        self.assertEqual(len(pending.to_specs()), len(plan))
        taken = store.take(approval_id)
        self.assertIsNotNone(taken)
        self.assertIsNone(store.get(approval_id))
        self.assertIsNone(store.take(approval_id))

    def test_list_returns_pending(self):
        store = ApprovalStore()
        for i in range(3):
            store.create(f"req-{i}", "svc", "unknown", [])
        self.assertEqual(len(store.list()), 3)


class TestEndToEnd(unittest.TestCase):
    def test_rca_category_to_remediation_plan(self):
        from src.rca import RCAClient

        evidence = {
            "service": "payment-api",
            "risk_score": 0.92,
            "risk_level": "CRITICAL",
            "failure_probability": 0.87,
            "components": {"failure_probability": 0.87, "error_rate": 0.9, "latency_trend": 0.4, "memory_growth": 0.3},
            "top_anomalous_features": [
                {"feature": "DB_usage_5min_avg", "zscore": 3.8, "direction": "up"},
                {"feature": "Error_rate", "zscore": 3.4, "direction": "up"},
            ],
            "metrics_deltas": {"Error_rate": 736.0},
            "correlated_logs": [
                {"level": "ERROR", "message": "connection pool exhausted - 0 available after acquire timeout"},
            ],
        }
        finding = RCAClient(provider="heuristic").analyze(evidence)
        self.assertEqual(finding.root_cause_category, "connection_pool_exhaustion")

        plan = plan_for_category(finding.root_cause_category, evidence["service"], context={"current_replicas": 1})
        executor = RemediationExecutor(backend=MockBackend(), mode="approval", audit_path=None)
        report = executor.execute(plan, approve=True)
        self.assertTrue(report.all_ok)
        self.assertIn("scale_replicas", [a.name for a in plan])


if __name__ == "__main__":
    unittest.main()