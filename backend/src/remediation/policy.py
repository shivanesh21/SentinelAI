from __future__ import annotations

from typing import Any, Iterable

from .model import ActionSpec

DEFAULT_WHITELIST = ("clear_cache",)
DEFAULT_MAX_RISK = "low"
DEFAULT_MAX_ACTIONS = 4


def policy_from_config(config: dict[str, Any] | None) -> "RemediationPolicy":
    """Build a policy from the `remediation.policy` settings block."""
    config = config or {}
    policy = config.get("remediation", {}).get("policy", {}) if "remediation" in config else config
    return RemediationPolicy(
        autonomous_whitelist=policy.get("autonomous_whitelist", DEFAULT_WHITELIST),
        max_risk_autonomous=policy.get("max_risk_autonomous", DEFAULT_MAX_RISK),
        max_actions_per_incident=policy.get("max_actions_per_incident", DEFAULT_MAX_ACTIONS),
    )


class RemediationPolicy:
    """Safety policy validated before any autonomous execution.

    - `autonomous_whitelist`: actions allowed to run without human sign-off.
    - `max_risk_autonomous`: highest risk level permitted autonomously.
    - `max_actions_per_incident`: hard cap on autonomously executed actions.
    """

    def __init__(
        self,
        autonomous_whitelist: Iterable[str] | None = None,
        max_risk_autonomous: str = DEFAULT_MAX_RISK,
        max_actions_per_incident: int = DEFAULT_MAX_ACTIONS,
    ):
        self.autonomous_whitelist = set(autonomous_whitelist or DEFAULT_WHITELIST)
        self.max_risk_autonomous = max_risk_autonomous or DEFAULT_MAX_RISK
        self.max_actions_per_incident = int(max_actions_per_incident or DEFAULT_MAX_ACTIONS)

    def allow_autonomous(self, spec: ActionSpec) -> tuple[bool, str]:
        """Return (allowed, reason) for an action under autonomous mode."""
        if spec.name not in self.autonomous_whitelist:
            return False, f"action '{spec.name}' not in autonomous whitelist: {sorted(self.autonomous_whitelist)}"
        if spec.risk not in (self.max_risk_autonomous,):
            return False, f"risk '{spec.risk}' exceeds autonomous limit '{self.max_risk_autonomous}'"
        return True, "ok"

    def decisions(self, plan: list[ActionSpec], autonomous: bool) -> list[tuple[ActionSpec, str, str]]:
        """Decide each action: ('execute', reason) | ('propose', reason) | ('skip', reason).

        Policy validation happens here, before any backend call. Actions are
        only auto-executed when whitelisted, within the risk budget, and within
        the max-actions-per-incident cap; everything else is deferred for
        human approval."""
        decisions: list[tuple[ActionSpec, str, str]] = []
        executed = 0
        for spec in plan:
            if not autonomous:
                decisions.append((spec, "propose", "requires approval"))
                continue
            if executed >= self.max_actions_per_incident:
                decisions.append(
                    (spec, "propose", f"exceeds max_actions_per_incident={self.max_actions_per_incident}")
                )
                continue
            allowed, reason = self.allow_autonomous(spec)
            if allowed:
                decisions.append((spec, "execute", reason))
                executed += 1
            else:
                decisions.append((spec, "propose", reason))
        return decisions

    def to_dict(self) -> dict[str, Any]:
        return {
            "autonomous_whitelist": sorted(self.autonomous_whitelist),
            "max_risk_autonomous": self.max_risk_autonomous,
            "max_actions_per_incident": self.max_actions_per_incident,
        }