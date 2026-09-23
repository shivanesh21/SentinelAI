from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .model import ActionSpec


@dataclass
class PendingApproval:
    approval_id: str
    request_id: str
    service: str
    category: str
    plan: list[dict[str, Any]]  # serialized ActionSpecs
    created_at: str
    context: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        request_id: str,
        service: str,
        category: str,
        plan: list[ActionSpec],
        context: dict[str, Any] | None = None,
    ) -> "PendingApproval":
        return cls(
            approval_id=uuid.uuid4().hex[:12],
            request_id=request_id,
            service=service,
            category=category,
            plan=[
                {
                    "name": spec.name,
                    "description": spec.description,
                    "params": spec.params,
                    "reversible": spec.reversible,
                    "risk": spec.risk,
                }
                for spec in plan
            ],
            created_at=datetime.now(timezone.utc).isoformat(),
            context=context or {},
        )

    def to_specs(self) -> list[ActionSpec]:
        return [ActionSpec(**item) for item in self.plan]

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__ | {"plan": list(self.plan)}


class ApprovalStore:
    """In-memory store of pending remediation plans awaiting human approval."""

    def __init__(self) -> None:
        self._items: dict[str, PendingApproval] = {}
        self._lock = threading.Lock()

    def create(
        self,
        request_id: str,
        service: str,
        category: str,
        plan: list[ActionSpec],
        context: dict[str, Any] | None = None,
    ) -> str:
        pending = PendingApproval.create(request_id, service, category, plan, context=context)
        with self._lock:
            self._items[pending.approval_id] = pending
        return pending.approval_id

    def get(self, approval_id: str) -> PendingApproval | None:
        with self._lock:
            return self._items.get(approval_id)

    def take(self, approval_id: str) -> PendingApproval | None:
        with self._lock:
            return self._items.pop(approval_id, None)

    def list(self) -> list[PendingApproval]:
        with self._lock:
            return [self._items[k] for k in sorted(self._items)]