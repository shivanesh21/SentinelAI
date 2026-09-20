from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Phase(str, Enum):
    HEALTHY = "healthy"
    DEGRADING = "degrading"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class ScenarioSpec:
    service: str
    scenario_type: str
    start_offset_min: float
    duration_min: float
    intensity: float = 0.7

    @property
    def end_offset_min(self) -> float:
        return self.start_offset_min + self.duration_min


@dataclass
class ScenarioEffect:
    overrides: dict[str, float] = field(default_factory=dict)
    phase: Phase = Phase.HEALTHY
    logs: list[tuple[str, str]] = field(default_factory=list)
    restarts: int = 0


def _lifecycle(ratio: float) -> tuple[Phase, str]:
    if ratio < 0.20:
        return Phase.DEGRADING, "degrading"
    if ratio < 0.85:
        return Phase.FAILED, "failed"
    return Phase.RECOVERING, "recovering"


def apply_scenario(
    scenario_type: str,
    elapsed_ratio: float,
    elapsed_min: float,
    intensity: float,
    current: dict[str, float],
) -> ScenarioEffect:
    c = current
    ratio = max(0.0, min(1.0, elapsed_ratio))
    effect = ScenarioEffect()
    phase, label = _lifecycle(ratio)

    if scenario_type == "memory_leak":
        growth = intensity * 30.0 * ratio
        memory = min(c["memory_util_pct"] + growth, 99.0)
        latency = c["latency_ms"] * (1.0 + 1.5 * growth / 30.0)
        overrides = {
            "memory_util_pct": memory,
            "latency_ms": latency,
            "cpu_util_pct": min(c["cpu_util_pct"] + 5.0 * ratio, 100.0),
        }
        if phase is Phase.DEGRADING:
            effect.logs.append(("WARNING", "memory usage increasing"))
        else:
            effect.logs.append(
                ("ERROR", "memory pressure critical: OOM risk detected")
            )
        effect.overrides = overrides
        effect.phase = phase
        return effect

    if scenario_type == "connection_pool_exhaustion":
        ramp = min(ratio / 0.6, 1.0)
        usage = 30.0 + 68.0 * ramp * intensity
        pool_size = max(c["db_connections"] / (max(c["db_connection_usage_pct"], 5.0) / 100.0), 50.0)
        latency = c["latency_ms"] * (1.0 + 2.5 * ramp)
        overrides = {
            "db_connection_usage_pct": min(usage, 100.0),
            "db_connections": pool_size * usage / 100.0,
            "latency_ms": latency,
            "http_5xx_rate": c["http_5xx_rate"] + 0.04 * ramp,
        }
        if usage > 80.0:
            effect.logs.extend(
                [
                    ("ERROR", "database connection timeout"),
                    ("ERROR", "connection pool exhausted"),
                    ("WARNING", "retry limit exceeded"),
                ]
            )
        elif usage > 55.0:
            effect.logs.append(("WARNING", "database connection usage rising"))
        effect.overrides = overrides
        effect.phase = phase
        return effect

    if scenario_type == "latency_spike":
        ramp = min(ratio / 0.15, 1.0)
        mult = 1.0 + 9.0 * intensity * ramp
        overrides = {
            "latency_ms": c["latency_ms"] * mult,
            "http_5xx_rate": c["http_5xx_rate"] + 0.06 * ramp,
            "request_rate_rps": c["request_rate_rps"] * (1.0 - 0.3 * ramp),
        }
        if ratio >= 0.95:
            effect.restarts = 1
        if phase is Phase.DEGRADING:
            effect.logs.append(("WARNING", "response latency increasing"))
        else:
            effect.logs.extend(
                [
                    ("ERROR", "service unavailable"),
                    ("WARNING", "retry attempt"),
                ]
            )
        effect.overrides = overrides
        effect.phase = phase
        return effect

    if scenario_type == "network_partition":
        wave = (math.sin(elapsed_min * 1.3) + 1.0) / 2.0
        partitioned = wave > 0.55
        if partitioned:
            overrides = {
                "request_rate_rps": c["request_rate_rps"] * (1.0 - wave),
                "latency_ms": c["latency_ms"] * 5.0,
                "http_5xx_rate": c["http_5xx_rate"] + 0.05,
                "http_4xx_rate": c["http_4xx_rate"] + 0.03,
                "network_in_mbps": c["network_in_mbps"] * 0.4,
                "network_out_mbps": c["network_out_mbps"] * 0.4,
            }
            effect.logs.extend(
                [
                    ("WARNING", "retry attempt"),
                    ("ERROR", "connection timeout"),
                ]
            )
        else:
            overrides = {"latency_ms": c["latency_ms"] * 2.0}
            effect.logs.append(("WARNING", "network flapping detected"))
        effect.overrides = overrides
        effect.phase = Phase.FAILED if partitioned else Phase.DEGRADING
        return effect

    if scenario_type == "disk_fill":
        ramp = min(ratio / 0.8, 1.0)
        disk = min(40.0 + 55.0 * ramp * intensity, 99.0)
        overrides = {
            "disk_util_pct": disk,
            "http_5xx_rate": c["http_5xx_rate"] + 0.05 * ramp,
            "latency_ms": c["latency_ms"] * (1.0 + 2.0 * ramp),
        }
        if disk > 90.0:
            effect.logs.append(("ERROR", "disk write failed: no space left"))
        elif disk > 60.0:
            effect.logs.append(("WARNING", "disk utilization high"))
        effect.overrides = overrides
        effect.phase = phase
        return effect

    return effect