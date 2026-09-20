from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .reader import (
    load_raw,
    logs_expected_columns,
    metrics_expected_columns,
    truth_expected_columns,
)
from .simulator.schema import METRIC_BOUNDS, METRIC_UNITS


@dataclass
class CheckResult:
    name: str
    status: str
    message: str
    details: Any = None


@dataclass
class ValidationReport:
    metric_rows: int = 0
    log_rows: int = 0
    incident_rows: int = 0
    interval_sec: int = 30
    checks: list[CheckResult] = field(default_factory=list)
    distributions: dict[str, Any] = field(default_factory=dict)

    def status(self) -> str:
        if any(c.status == "FAIL" for c in self.checks):
            return "FAIL"
        if any(c.status == "WARN" for c in self.checks):
            return "WARN"
        return "PASS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_rows": self.metric_rows,
            "log_rows": self.log_rows,
            "incident_rows": self.incident_rows,
            "interval_sec": self.interval_sec,
            "status": self.status(),
            "checks": [c.__dict__ for c in self.checks],
            "distributions": self.distributions,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


def _pass(name: str, message: str = "", details: Any = None) -> CheckResult:
    return CheckResult(name=name, status="PASS", message=message, details=details)


def _warn(name: str, message: str, details: Any = None) -> CheckResult:
    return CheckResult(name=name, status="WARN", message=message, details=details)


def _fail(name: str, message: str, details: Any = None) -> CheckResult:
    return CheckResult(name=name, status="FAIL", message=message, details=details)


def _info(name: str, message: str, details: Any = None) -> CheckResult:
    return CheckResult(name=name, status="INFO", message=message, details=details)


def _summary(df: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    summary = {}
    desc = df[cols].describe().T
    for col in cols:
        row = desc.loc[col]
        summary[col] = {
            "unit": METRIC_UNITS.get(col, ""),
            "count": int(desc.loc[col, "count"]),
            "mean": round(float(row["mean"]), 4),
            "std": round(float(row["std"]), 4),
            "min": round(float(row["min"]), 4),
            "p25": round(float(row["25%"]), 4),
            "p50": round(float(row["50%"]), 4),
            "p75": round(float(row["75%"]), 4),
            "max": round(float(row["max"]), 4),
        }
    return summary


def validate(root: str | Path, interval_sec: int = 30, schedule: list[dict] | None = None) -> ValidationReport:
    report = ValidationReport(interval_sec=interval_sec)
    data = load_raw(root)
    metrics = data["metrics"]
    logs = data["logs"]
    truth = data["truth"]

    if metrics.empty:
        report.checks.append(_fail("load_metrics", "no metrics found"))
        return report
    report.metric_rows = len(metrics)
    report.log_rows = len(logs)
    report.incident_rows = len(truth)

    metric_cols = [c for c in metrics_expected_columns() if c not in ("timestamp", "service", "healthy")]
    all_cols = metrics_expected_columns()

    schema_ok = list(metrics.columns) == all_cols
    report.checks.append(
        _pass("metrics_schema", f"{len(metrics.columns)} columns match TELEMETRY_HEADER")
        if schema_ok
        else _fail("metrics_schema", f"column mismatch: {list(metrics.columns)}")
    )
    report.checks.append(
        _pass("logs_schema", f"{len(logs.columns)} columns match LOG_HEADER")
        if list(logs.columns) == logs_expected_columns()
        else _fail("logs_schema", f"column mismatch: {list(logs.columns)}")
    )
    report.checks.append(
        _pass("truth_schema", f"{len(truth.columns)} columns match GROUND_TRUTH_HEADER")
        if list(truth.columns) == truth_expected_columns()
        else _fail("truth_schema", f"column mismatch: {list(truth.columns)}")
    )

    counts = metrics.groupby("service").size()
    counts_equal = counts.nunique() == 1
    total_span_min = (metrics["timestamp"].max() - metrics["timestamp"].min()).total_seconds() / 60.0
    expected_ticks = int(round((total_span_min + interval_sec / 60.0) * 60.0 / interval_sec))
    report.checks.append(
        _pass("per_service_ticks", f"all services have {int(counts.iloc[0])} rows" if counts_equal else f"counts: {counts.to_dict()}")
        if counts_equal
        else _fail("per_service_ticks", f"unequal tick counts across services: {counts.to_dict()}")
    )

    alignment_failures = 0
    ts_by_service = {s: sorted(v.to_list()) for s, v in metrics.groupby("service")["timestamp"]}
    first_ts = list(ts_by_service.values())[0]
    for service, ts in ts_by_service.items():
        if ts != first_ts:
            alignment_failures += 1
    report.checks.append(
        _pass("timestamp_alignment", f"{len(ts_by_service)} services share identical timestamps")
        if alignment_failures == 0
        else _fail("timestamp_alignment", f"{alignment_failures} services are misaligned")
    )

    if counts_equal:
        representative = ts_by_service[list(ts_by_service.keys())[0]]
        diffs = pd.Series(representative).diff().dropna().dt.total_seconds()
        spacing = sorted(diffs.unique())[:5] if len(diffs.unique()) > 5 else diffs.unique().tolist()
        bad = int((diffs - interval_sec).abs().gt(1).sum())
        report.checks.append(
            _pass("uniform_spacing", f"all {len(diffs)} gaps == {interval_sec}s (bad: {bad})")
            if bad == 0
            else _warn("uniform_spacing", f"{bad}/{len(diffs)} gaps deviate from {interval_sec}s; spacing={spacing}")
        )

    numeric = metrics[metric_cols].apply(pd.to_numeric, errors="coerce")
    missing = int(numeric.isna().sum().sum())
    infinities = int(numeric.isin([math.inf, -math.inf]).sum().sum())
    report.checks.append(
        _pass("missing_values", f"NaN={missing}, Inf={infinities} in numeric metrics")
        if missing == 0 and infinities == 0
        else _fail("missing_values", f"NaN={missing}, Inf={infinities}")
    )

    bound_hits: dict[str, int] = {}
    for col, (lo, hi) in METRIC_BOUNDS.items():
        if col not in numeric:
            continue
        if math.isfinite(hi):
            over = int((numeric[col] > hi).sum())
            under = int((numeric[col] < lo).sum())
            total = over + under
            if total:
                bound_hits[col] = total
    report.checks.append(
        _pass("metric_bounds", "all metrics within declared bounds")
        if not bound_hits
        else _fail("metric_bounds", f"out-of-bounds rows per metric: {bound_hits}")
    )

    healthy = metrics["healthy"].astype(bool) if "healthy" in metrics.columns else pd.Series(True, index=metrics.index)
    healthy_share = float(healthy.mean())
    per_service_unhealthy = metrics.assign(unhealthy=~healthy).groupby("service")["unhealthy"].sum()
    services_without_failure = per_service_unhealthy[per_service_unhealthy == 0].index.tolist()
    report.checks.append(
        _pass(
            "healthy_distribution",
            f"healthy={healthy_share:.3f}; every service shows degraded ticks" if not services_without_failure else f"healthy={healthy_share:.3f}"
        )
        if 0 < healthy_share < 1.0
        else _warn("healthy_distribution", f"healthy share {healthy_share:.3f} outside (0,1)")
    )
    if services_without_failure:
        report.checks.append(_info("per_service_failures", f"services never degraded (by design): {services_without_failure}"))

    valid_levels = set(logs["level"]) <= {"INFO", "WARNING", "ERROR"} if "level" in logs.columns else False
    report.checks.append(
        _pass("log_levels", f"levels={sorted(set(logs['level']))}")
        if valid_levels
        else _fail("log_levels", f"invalid levels: {set(logs['level']) - {'INFO', 'WARNING', 'ERROR'}}")
    )

    log_min, log_max = logs["timestamp"].min(), logs["timestamp"].max()
    met_min, met_max = metrics["timestamp"].min(), metrics["timestamp"].max()
    log_in_range = log_min >= met_min and log_max <= met_max
    report.checks.append(
        _pass("log_time_range", f"logs [{log_min}, {log_max}] within metrics [{met_min}, {met_max}]")
        if (not logs.empty and log_in_range)
        else _warn("log_time_range", "logs outside metrics time range or empty")
    )

    if truth.empty:
        report.checks.append(_fail("ground_truth", "no incidents recorded"))
    else:
        durations_min = (truth["end_ts"] - truth["start_ts"]).dt.total_seconds() / 60.0
        by_type = truth.groupby("scenario_type").size().to_dict()
        expected_types = {item["type"] for item in (schedule or [])}
        missing_schedule = expected_types - set(by_type)
        report.checks.append(
            _pass("incident_types", f"types seen: {by_type}")
            if not missing_schedule
            else _warn("incident_types", f"types expected but missing: {missing_schedule}")
        )
        negative = int((durations_min <= 0).sum())
        report.checks.append(
            _pass("incident_duration", f"all {len(durations_min)} incidents have positive duration; max={durations_min.max():.0f}min")
            if negative == 0
            else _fail("incident_duration", f"{negative} incidents have non-positive duration")
        )
        overlaps = 0
        for service, sub in truth.groupby("service"):
            merged = pd.DataFrame(
                {
                    "service": sub["service"],
                    "start": sub["start_ts"],
                    "end": sub["end_ts"],
                    "type": sub["scenario_type"],
                }
            ).sort_values("start")
            prev_end = None
            for _, row in merged.iterrows():
                if prev_end is not None and row["start"] < prev_end:
                    overlaps += 1
                prev_end = row["end"] if (prev_end is None or row["end"] > prev_end) else prev_end
        report.checks.append(
            _pass("incident_overlap", "no overlapping incidents within a service")
            if overlaps == 0
            else _warn("incident_overlap", f"{overlaps} overlapping incidents")
        )

        if schedule:
            mismatch = []
            for item in schedule:
                service, s_type = item["service"], item["type"]
                expected_dur = float(item["duration_min"])
                hits = truth[(truth["service"] == service) & (truth["scenario_type"] == s_type)]
                for _, row in hits.iterrows():
                    actual = (row["end_ts"] - row["start_ts"]).total_seconds() / 60.0
                    if abs(actual - expected_dur) > 3 * interval_sec / 60.0 * 2:
                        mismatch.append(f"{service}/{s_type}: expected {expected_dur:.0f}min got {actual:.1f}min")
            report.checks.append(
                _pass("incident_schedule_match", "incident durations match scheduled config +/-2 samples")
                if not mismatch
                else _warn("incident_schedule_match", "\n".join(mismatch))
            )

    healthy_series = metrics.set_index(["service", "timestamp"])["healthy"].astype(bool)
    if not truth.empty:
        windows = list(truth.itertuples(index=False))
        inside_hits = 0
        inside_total = 0
        outside_bad = 0
        for tup in windows:
            service, s_type, start, end = tup.service, tup.scenario_type, tup.start_ts, tup.end_ts
            mask = healthy_series.loc[service]
            seg = mask.loc[start:end]
            if len(seg) == 0:
                continue
            inside_total += len(seg)
            inside_hits += int((~seg).sum())
            after = mask.loc[end:]
            outside_bad += int(after[after.index <= end + pd.Timedelta(seconds=30 * interval_sec)].sum())
        inside_frac = inside_hits / max(inside_total, 1)
        report.checks.append(
            _pass("unhealthy_alignment", f"{inside_frac:.3f} of ticks inside incidents are degraded")
            if inside_frac >= 0.6
            else _warn("unhealthy_alignment", f"only {inside_frac:.3f} of incident ticks degraded")
        )
        recovery_ok = True
        for tup in windows:
            service, start, end = tup.service, tup.start_ts, tup.end_ts
            mask = healthy_series.loc[service]
            after_idx = mask.index[mask.index > end]
            if len(after_idx) < 1:
                report.checks.append(_info("recovery_after_incident", f"{service} incident ends at dataset end; no recovery window to verify"))
                continue
            after_5 = mask.loc[after_idx[0] : after_idx[min(len(after_idx) - 1, 4)]]
            if not after_5.all():
                recovery_ok = False
        report.checks.append(
            _pass("recovery_after_incident", "services return healthy after each incident")
            if recovery_ok
            else _warn("recovery_after_incident", "some services not healthy shortly after incident end")
        )

    if not logs.empty and not truth.empty:
        log_ts_indexed = logs.set_index("timestamp")
        correlated = 0
        for tup in truth.itertuples(index=False):
            seg = log_ts_indexed.loc[tup.start_ts:tup.end_ts]
            if not seg.empty and seg["level"].isin(["WARNING", "ERROR"]).any():
                correlated += 1
        report.checks.append(
            _pass("log_correlation", f"{correlated}/{len(truth)} incidents have WARNING/ERROR logs inside their window")
            if correlated == len(truth)
            else _warn("log_correlation", f"only {correlated}/{len(truth)} incidents correlated with logs")
        )

    report.checks.append(
        _info("expected_ticks", f"span {total_span_min:.1f} min @ {interval_sec}s -> ~{expected_ticks} ticks/service")
    )
    report.distributions = _summary(numeric, metric_cols)
    diurnal_cols = ["request_rate_rps", "latency_ms", "cpu_util_pct"]
    swings = {c: round(float(numeric[c].max() - numeric[c].min()), 2) for c in diurnal_cols}
    report.checks.append(_info("metric_range", f"{swings}"))
    report.checks.append(_info("incident_rows", f"{len(truth)} labelled incident windows"))

    return report