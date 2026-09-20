from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from ..log_generator import LogGenerator
from .failure_scenarios import ScenarioSpec, apply_scenario
from .schema import (
    GROUND_TRUTH_HEADER,
    LOG_HEADER,
    TELEMETRY_HEADER,
    IncidentRecord,
    LogLevel,
    LogRow,
    MetricRow,
    open_csv,
    open_jsonl,
    write_jsonl,
)
from .service import ServiceSimulator


@dataclass
class TickBatch:
    ts_iso: str
    metric_rows: list[MetricRow] = field(default_factory=list)
    log_rows: list[LogRow] = field(default_factory=list)
    incident_starts: list[IncidentRecord] = field(default_factory=list)
    incident_ends: list[IncidentRecord] = field(default_factory=list)


class TelemetryGenerator:
    def __init__(
        self,
        interval_sec: int = 30,
        seed: int = 42,
        start_ts: datetime | None = None,
    ) -> None:
        self.interval_sec = interval_sec
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.start_ts = start_ts or datetime.now(timezone.utc).replace(microsecond=0)

    def iter_steps(
        self,
        services: list[ServiceSimulator],
        scenarios: list[ScenarioSpec],
        duration_min: float | None = None,
    ):
        steps = None if duration_min is None else max(1, int(round(duration_min * 60.0 / self.interval_sec)))
        scenario_lut: dict[str, list[ScenarioSpec]] = {}
        for spec in scenarios:
            if duration_min is None or spec.start_offset_min < duration_min:
                scenario_lut.setdefault(spec.service, []).append(spec)

        log_gen = LogGenerator(self.seed)
        active_by_service: dict[str, dict[str, ScenarioSpec]] = {}
        open_incident: dict[str, IncidentRecord] = {}
        incident_seq = 0
        step = 0

        def step_once(offset_sec: float) -> TickBatch:
            nonlocal incident_seq
            nonlocal active_by_service
            current_ts = self.start_ts + timedelta(seconds=offset_sec)
            ts_iso = current_ts.isoformat()
            elapsed_hours = offset_sec / 3600.0
            now_active: dict[str, dict[str, ScenarioSpec]] = {}
            batch = TickBatch(ts_iso=ts_iso)

            for service in services:
                tick = getattr(service, "_tick", 0) + 1
                service._tick = tick
                metrics = service.baseline(elapsed_hours)
                scenario_effects = []

                for spec in scenario_lut.get(service.name, []):
                    if spec.start_offset_min * 60.0 <= offset_sec < spec.end_offset_min * 60.0:
                        now_active.setdefault(service.name, {})[spec.scenario_type] = spec
                        elapsed_min = offset_sec / 60.0 - spec.start_offset_min
                        ratio = elapsed_min / spec.duration_min
                        effect = apply_scenario(
                            spec.scenario_type,
                            ratio,
                            elapsed_min,
                            spec.intensity,
                            metrics,
                        )
                        scenario_effects.append(effect)
                        metrics.update(effect.overrides)
                        service.restart_count += effect.restarts

                healthy = all(
                    effect.phase not in ("degrading", "failed") for effect in scenario_effects
                )
                batch.metric_rows.append(
                    MetricRow(
                        timestamp=ts_iso,
                        service=service.name,
                        metric_values=metrics,
                        healthy=healthy,
                    )
                )

                if scenario_effects:
                    for effect in scenario_effects:
                        for level, message in effect.logs:
                            batch.log_rows.append(LogRow(ts_iso, service.name, LogLevel(level), message))
                else:
                    for level, message in log_gen.ambient(service.name, tick):
                        batch.log_rows.append(LogRow(ts_iso, service.name, LogLevel(level), message))

                previous = active_by_service.setdefault(service.name, {})
                for key, spec in now_active.get(service.name, {}).items():
                    if key not in previous:
                        incident_seq += 1
                        record = IncidentRecord(
                            incident_id=incident_seq,
                            service=service.name,
                            scenario_type=key,
                            start_ts=ts_iso,
                        )
                        open_incident[service.name] = record
                        batch.incident_starts.append(record)

                closed = open_incident.get(service.name)
                if closed and not now_active.get(service.name, {}):
                    closed.end_ts = ts_iso
                    batch.incident_ends.append(closed)
                    del open_incident[service.name]

            active_by_service = now_active
            return batch

        if steps is None:
            while True:
                yield step_once(step * self.interval_sec)
                step += 1

        for step in range(steps):
            yield step_once(step * self.interval_sec)

        for remaining in list(open_incident.values()):
            remaining.end_ts = remaining.start_ts
            final = TickBatch(ts_iso=remaining.start_ts)
            final.incident_ends.append(remaining)
            yield final
            open_incident.pop(remaining.service, None)

    def run(
        self,
        services: list[ServiceSimulator],
        scenarios: list[ScenarioSpec],
        duration_min: float,
        out_dir: Path,
        output_format: str = "csv",
    ) -> dict[str, int]:
        ext = "jsonl" if output_format == "jsonl" else "csv"
        metrics_path = out_dir / "raw" / "metrics" / f"metrics.{ext}"
        logs_path = out_dir / "raw" / "logs" / f"logs.{ext}"
        truth_path = out_dir / "raw" / "metrics" / f"ground_truth.{ext}"

        if output_format == "jsonl":
            metrics_handle, logs_handle, truth_handle = (
                open_jsonl(metrics_path),
                open_jsonl(logs_path),
                open_jsonl(truth_path),
            )
            metrics_writer = logs_writer = truth_writer = None
        else:
            metrics_handle, metrics_writer = open_csv(metrics_path, TELEMETRY_HEADER)
            logs_handle, logs_writer = open_csv(logs_path, LOG_HEADER)
            truth_handle, truth_writer = open_csv(truth_path, GROUND_TRUTH_HEADER)

        counts = {"rows": 0, "logs": 0, "incidents": 0}

        def emit_metrics(row: MetricRow) -> None:
            if output_format == "jsonl":
                write_jsonl(metrics_handle, row.to_dict())
            else:
                metrics_writer.writerow(list(row.to_dict().values()))

        def emit_log(row: LogRow) -> None:
            if output_format == "jsonl":
                write_jsonl(logs_handle, row.to_dict())
            else:
                logs_writer.writerow(list(row.to_dict().values()))

        def emit_truth(record: IncidentRecord) -> None:
            if output_format == "jsonl":
                write_jsonl(truth_handle, record.to_dict())
            else:
                truth_writer.writerow(list(record.to_dict().values()))

        for batch in self.iter_steps(services, scenarios, duration_min):
            for row in batch.metric_rows:
                emit_metrics(row)
                counts["rows"] += 1
            for row in batch.log_rows:
                emit_log(row)
                counts["logs"] += 1
            for record in batch.incident_ends:
                emit_truth(record)
                counts["incidents"] += 1

        metrics_handle.close()
        logs_handle.close()
        truth_handle.close()
        return counts