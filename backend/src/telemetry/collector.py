from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .simulator.failure_scenarios import ScenarioSpec
from .simulator.generator import TelemetryGenerator
from .simulator.schema import IncidentRecord
from .simulator.service import ServiceSimulator
from .storage import PartitionedStore


class TelemetryCollector:
    def __init__(
        self,
        store: PartitionedStore,
        interval_sec: int = 30,
        seed: int = 42,
        start_ts: datetime | None = None,
        live: bool = False,
    ) -> None:
        self.store = store
        self.interval_sec = interval_sec
        self.live = live
        self.generator = TelemetryGenerator(interval_sec=interval_sec, seed=seed, start_ts=start_ts)
        self._open_incidents: dict[str, IncidentRecord] = {}
        self._totals = {"rows": 0, "logs": 0, "incidents": 0}
        self._last_ts: str | None = None

    def _flush_open_incidents(self, end_ts: str) -> None:
        for service, record in self._open_incidents.items():
            record.end_ts = end_ts
        self.store.append_incidents(list(self._open_incidents.values()))
        self._open_incidents = {}

    def collect(
        self,
        services: list[ServiceSimulator],
        scenarios: list[ScenarioSpec],
        duration_min: float | None = None,
        stop: Any | None = None,
    ) -> dict[str, int]:
        try:
            for batch in self.generator.iter_steps(services, scenarios, duration_min):
                if stop is not None and stop.is_set():
                    break
                self.store.append_metrics(batch.metric_rows)
                self.store.append_logs(batch.log_rows)
                for record in batch.incident_starts:
                    self._open_incidents[record.service] = record
                for record in batch.incident_ends:
                    self.store.append_incidents([record])
                    if record.service in self._open_incidents:
                        del self._open_incidents[record.service]
                    self._totals["incidents"] += 1
                self._totals["rows"] += len(batch.metric_rows)
                self._totals["logs"] += len(batch.log_rows)
                self._last_ts = batch.ts_iso
                if self.live and duration_min is None:
                    time.sleep(self.interval_sec)
        except KeyboardInterrupt:
            self._flush_open_incidents(self._last_ts or self.generator.start_ts.isoformat())
            raise
        else:
            self._flush_open_incidents(self._last_ts or self.generator.start_ts.isoformat())
        return dict(self._totals)


def build_store(base_dir: Path, fmt: str = "csv", flush_every: int = 64) -> PartitionedStore:
    return PartitionedStore(base_dir=base_dir, fmt=fmt, flush_every=flush_every)