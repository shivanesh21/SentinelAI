from __future__ import annotations

import time
from typing import Any, Callable


class Backend:
    """Interface backends expose to the action library. All methods return a
    dict with at least ``ok`` and ``detail``; ``rollback_*`` methods restore
    the prior state for reversible actions."""

    def restart_container(self, container: str) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def clear_cache(self, service: str) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def scale_replicas(self, service: str, count: int) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def create_ticket(self, service: str, category: str, summary: str) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def rollback_restart(self, container: str) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def rollback_clear_cache(self, service: str) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def rollback_scale(self, service: str) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def rollback_ticket(self, ticket_id: str) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError


def _ok(detail: str, **extra: Any) -> dict[str, Any]:
    return {"ok": True, "detail": detail, **extra}


class MockBackend(Backend):
    """Deterministic, in-memory backend for safe defaults, demos, and tests.
    Every operation is simulated and fully reversible."""

    def __init__(self) -> None:
        self.containers: set[str] = set()
        self.replicas: dict[str, int] = {}
        self._last_scale: dict[str, int] = {}
        self.tickets: list[dict[str, Any]] = []

    def restart_container(self, container: str) -> dict[str, Any]:
        self.containers.add(container)
        return _ok(f"simulated restart of container '{container}'", container=container)

    def clear_cache(self, service: str) -> dict[str, Any]:
        return _ok(f"simulated cache eviction for '{service}'", service=service)

    def scale_replicas(self, service: str, count: int) -> dict[str, Any]:
        previous = self.replicas.get(service, 1)
        self._last_scale[service] = previous
        self.replicas[service] = int(count)
        return _ok(
            f"simulated replica count for '{service}' changed {previous} -> {count}",
            service=service,
            before=previous,
            after=int(count),
        )

    def create_ticket(self, service: str, category: str, summary: str) -> dict[str, Any]:
        ticket_id = f"INC-{len(self.tickets) + 1:04d}"
        self.tickets.append({"id": ticket_id, "service": service, "category": category, "summary": summary})
        return _ok(f"created incident ticket {ticket_id}", ticket_id=ticket_id)

    def rollback_restart(self, container: str) -> dict[str, Any]:
        self.containers.discard(container)
        return _ok(f"simulated rollback of restart for '{container}'", container=container)

    def rollback_clear_cache(self, service: str) -> dict[str, Any]:
        return _ok(f"simulated cache rebuild for '{service}'", service=service)

    def rollback_scale(self, service: str) -> dict[str, Any]:
        original = self._last_scale.get(service)
        if original is None:
            return {"ok": False, "detail": f"no recorded scale operation for '{service}'"}
        self.replicas[service] = original
        return _ok(f"simulated replica rollback for '{service}' to {original}", service=service, restored=original)

    def rollback_ticket(self, ticket_id: str) -> dict[str, Any]:
        return _ok(f"simulated close of ticket {ticket_id}", ticket_id=ticket_id)


class DockerBackend(MockBackend):
    """Real Docker backend used only when a docker client is importable.
    Falls back to simulated operations otherwise, keeping the library safe."""

    def __init__(self) -> None:
        super().__init__()
        self._docker = None
        try:
            import docker  # type: ignore

            self._docker = docker.from_env()
        except Exception as exc:  # pragma: no cover - environment-dependent
            self._unavailable = str(exc)

    def _client(self) -> Any:
        if self._docker is None:
            raise RuntimeError("docker SDK not available; using simulated backend")
        return self._docker

    def restart_container(self, container: str) -> dict[str, Any]:
        try:
            self._client().containers.get(container).restart()
            return _ok(f"restarted container '{container}' via Docker API")
        except Exception as exc:
            if getattr(self, "_unavailable", None):
                return super().restart_container(container)
            raise

    def scale_replicas(self, service: str, count: int) -> dict[str, Any]:
        return super().scale_replicas(service, count)