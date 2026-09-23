from __future__ import annotations

import time
from typing import Any, Callable

from .backends import Backend
from .model import ACTION_CLEAR_CACHE, ACTION_RESTART, ACTION_SCALE, ACTION_TICKET, ActionSpec, ActionResult

ActionRunner = Callable[[Backend, ActionSpec], dict[str, Any]]


def _run(backend: Backend, method: str, spec: ActionSpec) -> dict[str, Any]:
    fn = getattr(backend, method)
    if not callable(fn):
        return {"ok": False, "detail": f"backend has no {method}()"}
    return fn(**spec.params)


def _restart(backend: Backend, spec: ActionSpec) -> dict[str, Any]:
    return _run(backend, "restart_container", spec)


def _clear_cache(backend: Backend, spec: ActionSpec) -> dict[str, Any]:
    return _run(backend, "clear_cache", spec)


def _scale(backend: Backend, spec: ActionSpec) -> dict[str, Any]:
    return _run(backend, "scale_replicas", spec)


def _ticket(backend: Backend, spec: ActionSpec) -> dict[str, Any]:
    return _run(backend, "create_ticket", spec)


ACTION_REGISTRY: dict[str, dict[str, Any]] = {
    ACTION_RESTART: {
        "description": "Restart the affected container/service process",
        "risk": "medium",
        "reversible": True,
        "runner": _restart,
        "rollback": "rollback_restart",
        "rollback_params": ("container",),
    },
    ACTION_CLEAR_CACHE: {
        "description": "Evict cached state so the service rebuilds from source",
        "risk": "low",
        "reversible": True,
        "runner": _clear_cache,
        "rollback": "rollback_clear_cache",
        "rollback_params": ("service",),
    },
    ACTION_SCALE: {
        "description": "Adjust the replica count (scale up/down), tracked for rollback",
        "risk": "low",
        "reversible": True,
        "runner": _scale,
        "rollback": "rollback_scale",
        "rollback_params": ("service",),
    },
    ACTION_TICKET: {
        "description": "Create an incident ticket (Jira/Slack/mock) for human follow-up",
        "risk": "low",
        "reversible": False,
        "runner": _ticket,
        "rollback": "rollback_ticket",
        "rollback_params": ("ticket_id",),
    },
}


def execute_spec(backend: Backend, spec: ActionSpec, dry_run: bool) -> ActionResult:
    entry = ACTION_REGISTRY.get(spec.name)
    start = time.perf_counter()
    if entry is None:
        return ActionResult(name=spec.name, status="failed", detail=f"unknown action '{spec.name}'", params=spec.params)

    if dry_run:
        params = dict(spec.params)
        params.update({"description": entry["description"], "risk": entry["risk"]})
        return ActionResult(name=spec.name, status="planned", detail=entry["description"], params=params, duration_ms=0.0)

    try:
        outcome = entry["runner"](backend, spec)
        status = "ok" if outcome.get("ok") else "failed"
        params = dict(spec.params)
        params.update({k: v for k, v in outcome.items() if k not in ("ok", "detail")})
        return ActionResult(
            name=spec.name, status=status, detail=outcome.get("detail", ""), params=params,
            duration_ms=round((time.perf_counter() - start) * 1000, 3),
        )
    except Exception as exc:  # pragma: no cover - defensive
        return ActionResult(name=spec.name, status="failed", detail=str(exc), params=spec.params)


def rollback_spec(backend: Backend, spec: ActionSpec) -> ActionResult:
    entry = ACTION_REGISTRY.get(spec.name)
    if entry is None or not entry["reversible"]:
        return ActionResult(name=spec.name, status="skipped", detail="not reversible", params=spec.params)
    method = entry["rollback"]
    fn = getattr(backend, method)
    if not callable(fn):
        return ActionResult(name=spec.name, status="failed", detail=f"backend has no {method}()", params=spec.params)
    try:
        keys = entry.get("rollback_params", tuple(spec.params))
        outcome = fn(**{k: spec.params[k] for k in keys if k in spec.params})
        return ActionResult(
            name=spec.name, status="rolled_back" if outcome.get("ok") else "failed",
            detail=outcome.get("detail", ""), params=spec.params,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return ActionResult(name=spec.name, status="failed", detail=str(exc), params=spec.params)