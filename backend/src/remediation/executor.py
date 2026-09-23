from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .actions import execute_spec, rollback_spec
from .backends import Backend, MockBackend
from .model import DEFAULT_MODE, MODES, ActionSpec, ActionResult, ExecutionReport
from .policy import RemediationPolicy

DEFAULT_AUDIT_PATH = "data/remediation_audit.jsonl"


def resolve_mode(override: str | None = None) -> str:
    """Mode resolution order: explicit argument > REMEDIATION_MODE env > approval."""
    candidate = (override or os.environ.get("REMEDIATION_MODE") or DEFAULT_MODE).strip().lower()
    return candidate if candidate in MODES else DEFAULT_MODE


class RemediationExecutor:
    """Executes a remediation plan under a safety mode.

    - advisory   : recommendation only; never touches a backend.
    - approval   : human must approve (approve=True) before any execution;
                   otherwise returns the plan for sign-off.
    - autonomous : policy-validated execution — only whitelisted, low-risk
                   actions run; everything else is deferred for approval.
    """

    def __init__(
        self,
        backend: Backend | None = None,
        mode: str | None = None,
        audit_path: str | Path | None = DEFAULT_AUDIT_PATH,
        policy: RemediationPolicy | None = None,
    ):
        self.backend = backend or MockBackend()
        self.mode = resolve_mode(mode)
        self.audit_path = str(audit_path) if audit_path else None
        self.policy = policy or RemediationPolicy()

    def execute(
        self,
        plan: list[ActionSpec],
        approve: bool = False,
        context: dict[str, Any] | None = None,
    ) -> ExecutionReport:
        if self.mode == "advisory":
            return self._dry_execute(plan, context, status="planned")
        if self.mode == "autonomous":
            return self._autonomous_execute(plan, context)
        return self._approval_execute(plan, approve, context)

    def _dry_execute(self, plan: list[ActionSpec], context: dict[str, Any] | None, status: str) -> ExecutionReport:
        results = [execute_spec(self.backend, spec, dry_run=True) for spec in plan]
        for result in results:
            result.status = status
        report = ExecutionReport(mode=self.mode, dry_run=True, results=results, audit_path=self.audit_path)
        self._audit(report, context)
        return report

    def _approval_execute(self, plan: list[ActionSpec], approve: bool, context: dict[str, Any] | None) -> ExecutionReport:
        if not approve:
            return self._dry_execute(plan, context, status="planned")
        results = [execute_spec(self.backend, spec, dry_run=False) for spec in plan]
        report = ExecutionReport(mode=self.mode, dry_run=False, results=results, audit_path=self.audit_path)
        self._audit(report, context)
        return report

    def _autonomous_execute(self, plan: list[ActionSpec], context: dict[str, Any] | None) -> ExecutionReport:
        decisions = self.policy.decisions(plan, autonomous=True)
        results: list[ActionResult] = []
        for spec, decision, reason in decisions:
            if decision == "execute":
                results.append(execute_spec(self.backend, spec, dry_run=False))
            elif decision == "propose":
                results.append(
                    ActionResult(name=spec.name, status="requires_approval", detail=reason, params=spec.params)
                )
            else:
                results.append(
                    ActionResult(name=spec.name, status="skipped", detail=reason, params=spec.params)
                )
        report = ExecutionReport(mode=self.mode, dry_run=False, results=results, audit_path=self.audit_path)
        self._audit(report, context)
        return report

    def rollback(self, report: ExecutionReport) -> ExecutionReport:
        results = []
        for result in report.results:
            if result.status != "ok":
                results.append(ActionResult(name=result.name, status="skipped", detail="nothing to roll back"))
                continue
            spec = ActionSpec(name=result.name, description="rollback", params=result.params)
            results.append(rollback_spec(self.backend, spec))
        return ExecutionReport(mode=self.mode, dry_run=False, results=results, audit_path=self.audit_path)

    def _audit(self, report: ExecutionReport, context: dict[str, Any] | None) -> None:
        if not self.audit_path:
            return
        path = Path(self.audit_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "event_id": uuid.uuid4().hex[:12],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": report.mode,
            "dry_run": report.dry_run,
            "policy": self.policy.to_dict(),
            "context": context or {},
            "actions": [r.__dict__ for r in report.results],
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")