from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

METRIC_FIELDS: list[str] = [
    "cpu_util_pct",
    "memory_util_pct",
    "disk_util_pct",
    "network_in_mbps",
    "network_out_mbps",
    "request_rate_rps",
    "latency_ms",
    "http_4xx_rate",
    "http_5xx_rate",
    "db_connections",
    "db_connection_usage_pct",
    "container_restarts",
]

METRIC_UNITS: dict[str, str] = {
    "cpu_util_pct": "%",
    "memory_util_pct": "%",
    "disk_util_pct": "%",
    "network_in_mbps": "Mbps",
    "network_out_mbps": "Mbps",
    "request_rate_rps": "req/s",
    "latency_ms": "ms",
    "http_4xx_rate": "fraction",
    "http_5xx_rate": "fraction",
    "db_connections": "count",
    "db_connection_usage_pct": "%",
    "container_restarts": "count",
}

METRIC_BOUNDS: dict[str, tuple[float, float]] = {
    "cpu_util_pct": (0.0, 100.0),
    "memory_util_pct": (0.0, 100.0),
    "disk_util_pct": (0.0, 100.0),
    "network_in_mbps": (0.0, float("inf")),
    "network_out_mbps": (0.0, float("inf")),
    "request_rate_rps": (0.0, float("inf")),
    "latency_ms": (0.0, float("inf")),
    "http_4xx_rate": (0.0, 1.0),
    "http_5xx_rate": (0.0, 1.0),
    "db_connections": (0.0, float("inf")),
    "db_connection_usage_pct": (0.0, 100.0),
    "container_restarts": (0.0, float("inf")),
}

TELEMETRY_HEADER: list[str] = ["timestamp", "service"] + METRIC_FIELDS + ["healthy"]
LOG_HEADER: list[str] = ["timestamp", "service", "level", "message"]
GROUND_TRUTH_HEADER: list[str] = [
    "incident_id",
    "service",
    "scenario_type",
    "start_ts",
    "end_ts",
]


class LogLevel(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class MetricRow:
    timestamp: str
    service: str
    metric_values: dict[str, float]
    healthy: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "service": self.service,
            **self.metric_values,
            "healthy": self.healthy,
        }


@dataclass
class LogRow:
    timestamp: str
    service: str
    level: LogLevel
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "service": self.service,
            "level": self.level.value,
            "message": self.message,
        }


@dataclass
class IncidentRecord:
    incident_id: int
    service: str
    scenario_type: str
    start_ts: str
    end_ts: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "service": self.service,
            "scenario_type": self.scenario_type,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
        }


def open_csv(path: Path, header: list[str]) -> tuple[Any, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(header)
    return handle, writer


def open_jsonl(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def write_jsonl(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record) + "\n")