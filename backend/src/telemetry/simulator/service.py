from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .schema import METRIC_FIELDS

DEFAULT_PROFILE: dict[str, float] = {
    "base_cpu": 32.0,
    "base_mem": 45.0,
    "base_disk": 40.0,
    "base_net": 20.0,
    "base_req_rps": 150.0,
    "base_latency_ms": 120.0,
    "base_4xx": 0.004,
    "base_5xx": 0.001,
    "base_db_connections": 40.0,
    "base_db_usage_pct": 30.0,
}


@dataclass
class ServiceSimulator:
    name: str
    replicas: int = 1
    profile: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        merged = dict(DEFAULT_PROFILE)
        merged.update(self.profile)
        self.profile = merged
        self._phase = math.sin(2 * math.pi * (hash(self.name) % 24) / 24.0)
        self.restart_count: int = 0
        self._rng = np.random.default_rng(abs(hash(self.name)) % (2**32))

    def baseline(self, elapsed_hours: float) -> dict[str, float]:
        p = self.profile
        diurnal = 1.0 + 0.35 * math.sin(2 * math.pi * (elapsed_hours - 6.0) / 24.0)
        load = diurnal * (0.85 + 0.15 * self._phase)
        rng = self._rng

        cpu = min(max(p["base_cpu"] * (0.8 + 0.4 * (diurnal - 1.0)) + rng.normal(0, 3.0), 0.0), 100.0)
        memory = min(max(p["base_mem"] + 0.02 * elapsed_hours + rng.normal(0, 0.8), 0.0), 100.0)
        disk = min(max(p["base_disk"] + 0.01 * elapsed_hours + rng.normal(0, 0.2), 0.0), 100.0)
        net_in = max(p["base_net"] * load + rng.normal(0, 2.0), 0.0)
        net_out = max(p["base_net"] * 0.4 * load + rng.normal(0, 1.0), 0.0)
        request_rate = max(p["base_req_rps"] * load + rng.normal(0, 8.0), 0.0)
        latency = max(p["base_latency_ms"] * (0.85 + 0.3 * (diurnal - 1.0) + 0.3 * (cpu - p["base_cpu"]) / 100.0) + rng.normal(0, 6.0), 1.0)
        http_4xx = max(p["base_4xx"] * load + rng.normal(0, 0.0004), 0.0)
        http_5xx = max(p["base_5xx"] * load + rng.normal(0, 0.0002), 0.0)
        db_connections = max(p["base_db_connections"] * (load * 0.7 + 0.3) + rng.normal(0, 2.0), 0.0)
        db_usage = min(max(p["base_db_usage_pct"] + 10.0 * (load - 1.0) + rng.normal(0, 1.5), 0.0), 100.0)

        values = {
            "cpu_util_pct": cpu,
            "memory_util_pct": memory,
            "disk_util_pct": disk,
            "network_in_mbps": net_in,
            "network_out_mbps": net_out,
            "request_rate_rps": request_rate,
            "latency_ms": latency,
            "http_4xx_rate": http_4xx,
            "http_5xx_rate": http_5xx,
            "db_connections": db_connections,
            "db_connection_usage_pct": db_usage,
            "container_restarts": float(self.restart_count),
        }
        return {k: float(values[k]) for k in METRIC_FIELDS}