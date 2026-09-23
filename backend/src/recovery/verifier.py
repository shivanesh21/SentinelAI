from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

DEFAULT_METRICS = ["Error_rate", "Latency_15min_avg", "Memory_growth_rate"]
DEFAULT_THRESHOLDS = {
    "Error_rate": 0.02,
    "Latency_15min_avg": 300.0,
    "Memory_growth_rate": 0.05,
}
DEFAULT_COOLDOWN_SEC = 60
DEFAULT_MIN_IMPROVEMENT = 0.05
DEFAULT_LOG_PATH = "data/recovery_log.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MetricCheck:
    name: str
    before: float
    after: float
    delta_pct: float
    threshold: Optional[float]
    status: str
    recovered: bool

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "before": self.before,
            "after": self.after,
            "delta_pct": round(self.delta_pct, 4),
            "threshold": self.threshold,
            "status": self.status,
            "recovered": self.recovered,
        }


@dataclass
class RecoveryReport:
    service: str
    status: str
    recovery_time_sec: float
    cooldown_sec: int
    metrics: List[MetricCheck] = field(default_factory=list)
    checked_at: str = field(default_factory=_now)
    action: Optional[str] = None
    before_rows: int = 0
    after_rows: int = 0
    log_path: Optional[str] = None

    @property
    def recovered_metrics(self) -> List[str]:
        return [m.name for m in self.metrics if m.recovered]

    def to_dict(self) -> dict:
        return {
            "service": self.service,
            "status": self.status,
            "recovery_time_sec": round(self.recovery_time_sec, 3),
            "cooldown_sec": self.cooldown_sec,
            "metrics": [m.to_dict() for m in self.metrics],
            "checked_at": self.checked_at,
            "action": self.action,
            "before_rows": self.before_rows,
            "after_rows": self.after_rows,
            "log_path": self.log_path,
        }


class RecoveryVerifier:
    def __init__(
        self,
        cooldown_sec: int = DEFAULT_COOLDOWN_SEC,
        metrics: Optional[List[str]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        min_improvement: float = DEFAULT_MIN_IMPROVEMENT,
        log_path: str = DEFAULT_LOG_PATH,
    ) -> None:
        self.cooldown_sec = cooldown_sec
        self.metrics = list(metrics or DEFAULT_METRICS)
        self.thresholds = dict(thresholds or DEFAULT_THRESHOLDS)
        self.min_improvement = min_improvement
        self.log_path = log_path

    def snapshot(self, frame: pd.DataFrame, service: str) -> Dict[str, float]:
        rows = frame
        if "service" in frame.columns:
            rows = frame[frame["service"] == service]
        if rows.empty:
            raise ValueError(f"no rows found for service {service!r}")
        available = [m for m in self.metrics if m in rows.columns]
        if not available:
            raise ValueError(f"none of the configured metrics appear in frame ({self.metrics})")
        return {m: float(rows[m].mean()) for m in available}

    @staticmethod
    def _ts_mid(frame: pd.DataFrame) -> Optional[pd.Timestamp]:
        if "timestamp" not in frame.columns:
            return None
        try:
            ts = pd.to_datetime(frame["timestamp"])
        except Exception:
            return None
        if ts.empty:
            return None
        return ts.median()

    def _check_metric(self, metric: str, before: float, after: float) -> MetricCheck:
        delta_pct = ((after - before) / before * 100.0) if before else 0.0
        threshold = self.thresholds.get(metric)
        if threshold is not None and after <= threshold:
            return MetricCheck(metric, before, after, delta_pct, threshold, "healthy", True)
        if after <= before * (1.0 - self.min_improvement):
            return MetricCheck(metric, before, after, delta_pct, threshold, "improved", True)
        if after > before * (1.0 + self.min_improvement):
            return MetricCheck(metric, before, after, delta_pct, threshold, "worsened", False)
        return MetricCheck(metric, before, after, delta_pct, threshold, "unchanged", False)

    def verify_snapshots(
        self,
        before: Dict[str, float],
        after: Dict[str, float],
        service: str,
        action: Optional[str] = None,
        before_rows: int = 0,
        after_rows: int = 0,
    ) -> RecoveryReport:
        checks: List[MetricCheck] = [
            self._check_metric(metric, before[metric], after[metric])
            for metric in self.metrics
            if metric in before and metric in after
        ]
        status = "SUCCESS" if checks and all(c.recovered for c in checks) else "FAILURE"
        return RecoveryReport(
            service=service,
            status=status,
            recovery_time_sec=float(self.cooldown_sec),
            cooldown_sec=self.cooldown_sec,
            metrics=checks,
            action=action,
            before_rows=before_rows,
            after_rows=after_rows,
            log_path=self.log_path,
        )

    def verify(
        self,
        before_frame: pd.DataFrame,
        after_frame: pd.DataFrame,
        service: str,
        action: Optional[str] = None,
    ) -> RecoveryReport:
        before = self.snapshot(before_frame, service)
        after = self.snapshot(after_frame, service)
        report = self.verify_snapshots(
            before,
            after,
            service,
            action=action,
            before_rows=len(before_frame),
            after_rows=len(after_frame),
        )
        mid_before = self._ts_mid(before_frame)
        mid_after = self._ts_mid(after_frame)
        if mid_before is not None and mid_after is not None:
            report.recovery_time_sec = max(
                float((mid_after - mid_before).total_seconds()), float(self.cooldown_sec)
            )
        return report

    def log(self, report: RecoveryReport) -> str:
        path = Path(report.log_path or self.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(report.to_dict())
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return line

    def read(self, limit: int = 20) -> List[dict]:
        path = Path(self.log_path)
        if not path.exists():
            return []
        records = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if limit <= 0:
            return []
        records = records[-limit:]
        return records


def verify_after_remediation(
    execution_report,
    before_frame: pd.DataFrame,
    after_frame: pd.DataFrame,
    service: str,
    action: Optional[str] = None,
    verifier: Optional[RecoveryVerifier] = None,
) -> Optional[RecoveryReport]:
    verifier = verifier or RecoveryVerifier()
    if getattr(execution_report, "executed", 0) == 0:
        return None
    return verifier.verify(before_frame, after_frame, service, action=action)