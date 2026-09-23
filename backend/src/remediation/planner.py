from __future__ import annotations

from typing import Any

from ..rca.schema import CAUSE_CATEGORIES
from .actions import ACTION_REGISTRY
from .model import ACTION_CLEAR_CACHE, ACTION_RESTART, ACTION_SCALE, ACTION_TICKET, ActionSpec

ROOT_CAUSE_ACTION_MAP: dict[str, list[dict[str, Any]]] = {
    # ordered: highest-value / most direct action first
    "memory_leak": [
        {"name": ACTION_RESTART, "description": "Roll-restart the service to reclaim leaked memory"},
        {"name": ACTION_TICKET, "description": "Open a memory-leak diagnosis ticket (heap dump + profile)"},
    ],
    "connection_pool_exhaustion": [
        {"name": ACTION_SCALE, "description": "Scale up replicas to relieve pool pressure", "params": {"count": 2}},
        {"name": ACTION_CLEAR_CACHE, "description": "Flush local caches holding pooled connections"},
        {"name": ACTION_TICKET, "description": "Open a connection-pool tuning ticket"},
    ],
    "latency_spike": [
        {"name": ACTION_SCALE, "description": "Scale up replicas to cut queue depth", "params": {"count": 2}},
        {"name": ACTION_CLEAR_CACHE, "description": "Clear hot-path cache to refresh stale entries"},
        {"name": ACTION_TICKET, "description": "Open latency/SLO breach ticket"},
    ],
    "network_partition": [
        {"name": ACTION_RESTART, "description": "Restart the service to re-establish connectivity"},
        {"name": ACTION_TICKET, "description": "Open network-partition ticket for mesh/firewall check"},
    ],
    "disk_fill": [
        {"name": ACTION_CLEAR_CACHE, "description": "Free disk by evicting temp/cache data"},
        {"name": ACTION_RESTART, "description": "Restart the service to release file handles and temp files"},
        {"name": ACTION_TICKET, "description": "Open disk-capacity ticket (cleanup + alarm)"},
    ],
    "code_deployment": [
        {"name": ACTION_RESTART, "description": "Roll back the failed deployment by restarting prior image"},
        {"name": ACTION_TICKET, "description": "Open deployment-regression ticket"},
    ],
    "traffic_spike": [
        {"name": ACTION_SCALE, "description": "Scale up replicas to absorb traffic burst", "params": {"count": 2}},
        {"name": ACTION_CLEAR_CACHE, "description": "Validate cached responses for burst traffic"},
        {"name": ACTION_TICKET, "description": "Open traffic-spike ticket"},
    ],
    "unknown": [
        {"name": ACTION_TICKET, "description": "Open investigatory ticket for unknown root cause"},
    ],
}


class RemediationPlanner:
    """Maps a root-cause category to a list of safe remediation actions."""

    def __init__(self, action_map: dict[str, list[dict[str, Any]]] | None = None):
        self.action_map = action_map if action_map is not None else ROOT_CAUSE_ACTION_MAP

    def plan_for_category(
        self,
        category: str,
        service: str,
        context: dict[str, Any] | None = None,
    ) -> list[ActionSpec]:
        """Return an ordered plan of ActionSpecs for a root-cause category."""
        category = category if category in self.action_map else "unknown"
        template = self.action_map[category]
        context = context or {}
        specs = []
        for item in template:
            name = item["name"]
            params = dict(item.get("params", {}))
            entry = ACTION_REGISTRY.get(name, {})
            if name == ACTION_SCALE and "count" in params:
                delta = int(params["count"])
                params["count"] = int(context.get("current_replicas", 1)) + delta
            if name == ACTION_RESTART:
                params.setdefault("container", service)
            else:
                params.setdefault("service", service)
            if name == ACTION_TICKET:
                params.setdefault("category", category)
                params.setdefault("summary", f"[{service}] {category.replace('_', ' ')} detected")
            specs.append(
                ActionSpec(
                    name=name,
                    description=item.get("description", entry.get("description", name)),
                    params=params,
                    reversible=bool(entry.get("reversible", True)),
                    risk=str(entry.get("risk", "low")),
                )
            )
        return specs


def plan_for_category(
    category: str,
    service: str,
    context: dict[str, Any] | None = None,
    planner: RemediationPlanner | None = None,
) -> list[ActionSpec]:
    return (planner or RemediationPlanner()).plan_for_category(category, service, context)