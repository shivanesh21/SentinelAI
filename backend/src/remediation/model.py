from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ACTION_RESTART = "restart_container"
ACTION_CLEAR_CACHE = "clear_cache"
ACTION_SCALE = "scale_replicas"
ACTION_TICKET = "create_incident_ticket"

MODES = ("advisory", "approval", "autonomous")
DEFAULT_MODE = "approval"

ACTION_NAMES = (ACTION_RESTART, ACTION_CLEAR_CACHE, ACTION_SCALE, ACTION_TICKET)


@dataclass
class ActionSpec:
    """A single, safe, typically reversible remediation action."""

    name: str
    description: str
    params: dict[str, Any] = field(default_factory=dict)
    reversible: bool = True
    risk: str = "low"  # low | medium | high


@dataclass
class ActionResult:
    name: str
    status: str  # planned | simulated | ok | failed | rolled_back | skipped
    detail: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0


@dataclass
class ExecutionReport:
    mode: str
    dry_run: bool
    results: list[ActionResult] = field(default_factory=list)
    audit_path: str | None = None

    @property
    def executed(self) -> int:
        return sum(1 for r in self.results if r.status == "ok")

    @property
    def pending(self) -> int:
        return sum(1 for r in self.results if r.status == "requires_approval")

    @property
    def all_ok(self) -> bool:
        return bool(self.results) and all(r.status == "ok" for r in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "dry_run": self.dry_run,
            "audit_path": self.audit_path,
            "actions": [r.__dict__ for r in self.results],
        }