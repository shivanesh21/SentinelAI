from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def calculate_psi(
    reference: Sequence[float] | np.ndarray | pd.Series,
    target: Sequence[float] | np.ndarray | pd.Series,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calculate the Population Stability Index (PSI) between reference and target distributions.

    PSI < 0.10: No significant distribution change (stable)
    0.10 <= PSI < 0.25: Moderate distribution drift
    PSI >= 0.25: Significant distribution drift (retraining suggested)
    """
    ref_arr = np.asarray(reference, dtype=float)
    target_arr = np.asarray(target, dtype=float)

    ref_clean = ref_arr[~np.isnan(ref_arr)]
    target_clean = target_arr[~np.isnan(target_arr)]

    if len(ref_clean) == 0 or len(target_clean) == 0:
        return 0.0

    # If reference has no variance
    ref_min, ref_max = np.min(ref_clean), np.max(ref_clean)
    if np.isclose(ref_min, ref_max):
        target_min, target_max = np.min(target_clean), np.max(target_clean)
        if np.isclose(ref_min, target_min) and np.isclose(ref_max, target_max):
            return 0.0
        return 1.0  # complete shift away from constant baseline

    # Compute quantile bin edges based on reference
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref_clean, percentiles)

    # In case of duplicate quantiles (e.g., highly repeated zero values)
    unique_edges = np.unique(bin_edges)
    if len(unique_edges) < 2:
        bin_edges = np.linspace(ref_min, ref_max, num_bins + 1)
    elif len(unique_edges) < len(bin_edges):
        # Fallback to unique edges if quantiles collapse
        bin_edges = unique_edges

    # Extend boundary edges to [-inf, +inf] to encompass all target values
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    target_counts, _ = np.histogram(target_clean, bins=bin_edges)

    # Proportions with epsilon smoothing to prevent division by zero or log(0)
    k = len(ref_counts)
    ref_prop = (ref_counts + epsilon) / (len(ref_clean) + epsilon * k)
    target_prop = (target_counts + epsilon) / (len(target_clean) + epsilon * k)

    psi_value = np.sum((target_prop - ref_prop) * np.log(target_prop / ref_prop))
    return float(max(0.0, psi_value))


def calculate_ks_test(
    reference: Sequence[float] | np.ndarray | pd.Series,
    target: Sequence[float] | np.ndarray | pd.Series,
) -> tuple[float, float]:
    """Calculate the two-sample Kolmogorov-Smirnov test statistic and p-value.
    A small p-value (< alpha) indicates that the two samples are drawn from different distributions.
    """
    ref_arr = np.asarray(reference, dtype=float)
    target_arr = np.asarray(target, dtype=float)

    ref_clean = ref_arr[~np.isnan(ref_arr)]
    target_clean = target_arr[~np.isnan(target_arr)]

    if len(ref_clean) == 0 or len(target_clean) == 0:
        return 0.0, 1.0

    # If both are constant
    if np.allclose(ref_clean, ref_clean[0]) and np.allclose(target_clean, target_clean[0]):
        if np.isclose(ref_clean[0], target_clean[0]):
            return 0.0, 1.0
        return 1.0, 0.0

    res = stats.ks_2samp(ref_clean, target_clean)
    return float(res.statistic), float(res.pvalue)


@dataclass
class FeatureDriftResult:
    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    drift_level: str  # "none", "moderate", "significant"
    drift_detected: bool
    reference_mean: float
    target_mean: float
    mean_shift: float
    reference_std: float
    target_std: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DataDriftSummary:
    total_features: int
    drifted_features_count: int
    drift_share: float
    dataset_drift: bool
    drifted_features: list[str] = field(default_factory=list)
    moderate_drift_features: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PerformanceMetrics:
    sample_count: int
    f1: float
    precision: float
    recall: float
    accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PerformanceReport:
    baseline_f1: float | None
    rolling_metrics: PerformanceMetrics | None
    f1_drop: float | None
    degraded: bool
    status: str  # "ok", "degraded", "insufficient_data"

    def to_dict(self) -> dict[str, Any]:
        d = {
            "baseline_f1": self.baseline_f1,
            "rolling_metrics": self.rolling_metrics.to_dict() if self.rolling_metrics else None,
            "f1_drop": self.f1_drop,
            "degraded": self.degraded,
            "status": self.status,
        }
        return d


@dataclass
class RetrainingTriggerResult:
    retraining_required: bool
    trigger_reasons: list[str] = field(default_factory=list)
    severity: str = "normal"  # "normal", "warning", "critical"
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DriftMonitoringReport:
    timestamp: str
    reference_samples: int
    target_samples: int
    data_drift: DataDriftSummary
    feature_metrics: dict[str, FeatureDriftResult]
    performance: PerformanceReport
    trigger: RetrainingTriggerResult
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "reference_samples": self.reference_samples,
            "target_samples": self.target_samples,
            "data_drift": self.data_drift.to_dict(),
            "feature_metrics": {k: v.to_dict() for k, v in self.feature_metrics.items()},
            "performance": self.performance.to_dict(),
            "trigger": self.trigger.to_dict(),
            "metadata": self.metadata,
        }

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        payload = json.dumps(self.to_dict(), indent=indent, default=str)
        if path:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(payload, encoding="utf-8")
        return payload


class DriftDetector:
    """Detects feature-level and dataset-level distribution drift using PSI and KS-test."""

    def __init__(
        self,
        num_bins: int = 10,
        moderate_psi: float = 0.10,
        significant_psi: float = 0.25,
        ks_alpha: float = 0.05,
        dataset_drift_share_threshold: float = 0.30,
    ) -> None:
        self.num_bins = num_bins
        self.moderate_psi = moderate_psi
        self.significant_psi = significant_psi
        self.ks_alpha = ks_alpha
        self.dataset_drift_share_threshold = dataset_drift_share_threshold

    def analyze_feature(
        self,
        reference: Sequence[float] | np.ndarray | pd.Series,
        target: Sequence[float] | np.ndarray | pd.Series,
        feature_name: str = "feature",
    ) -> FeatureDriftResult:
        ref_arr = np.asarray(reference, dtype=float)
        target_arr = np.asarray(target, dtype=float)
        ref_clean = ref_arr[~np.isnan(ref_arr)]
        target_clean = target_arr[~np.isnan(target_arr)]

        psi = calculate_psi(ref_clean, target_clean, num_bins=self.num_bins)
        ks_stat, ks_pval = calculate_ks_test(ref_clean, target_clean)

        ref_mean = float(np.mean(ref_clean)) if len(ref_clean) else 0.0
        tgt_mean = float(np.mean(target_clean)) if len(target_clean) else 0.0
        ref_std = float(np.std(ref_clean)) if len(ref_clean) else 0.0
        tgt_std = float(np.std(target_clean)) if len(target_clean) else 0.0

        if psi >= self.significant_psi:
            drift_level = "significant"
            drift_detected = True
        elif psi >= self.moderate_psi:
            drift_level = "moderate"
            drift_detected = ks_pval < self.ks_alpha
        else:
            drift_level = "none"
            drift_detected = False

        return FeatureDriftResult(
            feature=feature_name,
            psi=round(psi, 4),
            ks_statistic=round(ks_stat, 4),
            ks_pvalue=round(ks_pval, 6),
            drift_level=drift_level,
            drift_detected=drift_detected,
            reference_mean=round(ref_mean, 4),
            target_mean=round(tgt_mean, 4),
            mean_shift=round(tgt_mean - ref_mean, 4),
            reference_std=round(ref_std, 4),
            target_std=round(tgt_std, 4),
        )

    def analyze(
        self,
        reference_df: pd.DataFrame,
        target_df: pd.DataFrame,
        feature_columns: list[str] | None = None,
    ) -> tuple[DataDriftSummary, dict[str, FeatureDriftResult]]:
        if feature_columns is None:
            # Select numeric columns present in both frames, excluding metadata/labels
            exclude = {"timestamp", "service", "failure_in_next_10min", "in_failure", "time_to_failure_min"}
            feature_columns = [
                c
                for c in reference_df.columns
                if c in target_df.columns
                and c not in exclude
                and pd.api.types.is_numeric_dtype(reference_df[c])
            ]

        results: dict[str, FeatureDriftResult] = {}
        drifted = []
        moderate = []

        for col in feature_columns:
            res = self.analyze_feature(reference_df[col], target_df[col], feature_name=col)
            results[col] = res
            if res.drift_level == "significant":
                drifted.append(col)
            elif res.drift_level == "moderate":
                moderate.append(col)

        total = len(feature_columns)
        drift_share = (len(drifted) / total) if total > 0 else 0.0
        dataset_drift = drift_share >= self.dataset_drift_share_threshold

        summary = DataDriftSummary(
            total_features=total,
            drifted_features_count=len(drifted),
            drift_share=round(drift_share, 4),
            dataset_drift=dataset_drift,
            drifted_features=drifted,
            moderate_drift_features=moderate,
        )
        return summary, results


class PerformanceMonitor:
    """Monitors rolling model classification metrics on recent labeled data and detects degradation."""

    def __init__(
        self,
        f1_drop_threshold: float = 0.15,
        min_f1_threshold: float = 0.60,
        min_samples: int = 50,
    ) -> None:
        self.f1_drop_threshold = f1_drop_threshold
        self.min_f1_threshold = min_f1_threshold
        self.min_samples = min_samples

    def compute_metrics(
        self,
        y_true: Sequence[int] | np.ndarray | pd.Series,
        y_pred: Sequence[int] | np.ndarray | pd.Series,
    ) -> PerformanceMetrics:
        yt = np.asarray(y_true, dtype=int)
        yp = np.asarray(y_pred, dtype=int)
        n = len(yt)
        if n == 0:
            return PerformanceMetrics(sample_count=0, f1=0.0, precision=0.0, recall=0.0, accuracy=0.0)

        f1 = float(f1_score(yt, yp, zero_division=0))
        prec = float(precision_score(yt, yp, zero_division=0))
        rec = float(recall_score(yt, yp, zero_division=0))
        acc = float(accuracy_score(yt, yp))

        return PerformanceMetrics(
            sample_count=n,
            f1=round(f1, 4),
            precision=round(prec, 4),
            recall=round(rec, 4),
            accuracy=round(acc, 4),
        )

    def evaluate(
        self,
        y_true: Sequence[int] | np.ndarray | pd.Series,
        y_pred: Sequence[int] | np.ndarray | pd.Series,
        baseline_f1: float | None = None,
    ) -> PerformanceReport:
        metrics = self.compute_metrics(y_true, y_pred)
        if metrics.sample_count < self.min_samples:
            return PerformanceReport(
                baseline_f1=baseline_f1,
                rolling_metrics=metrics,
                f1_drop=None,
                degraded=False,
                status="insufficient_data",
            )

        f1_drop = None
        degraded = False
        if baseline_f1 is not None:
            f1_drop = round(baseline_f1 - metrics.f1, 4)
            if f1_drop >= self.f1_drop_threshold or metrics.f1 < self.min_f1_threshold:
                degraded = True
        elif metrics.f1 < self.min_f1_threshold:
            degraded = True

        status = "degraded" if degraded else "ok"
        return PerformanceReport(
            baseline_f1=baseline_f1,
            rolling_metrics=metrics,
            f1_drop=f1_drop,
            degraded=degraded,
            status=status,
        )


class RetrainingTrigger:
    """Evaluates data drift and model degradation to determine if model retraining is required."""

    def __init__(
        self,
        drift_share_threshold: float = 0.30,
        critical_features: Sequence[str] | None = None,
    ) -> None:
        self.drift_share_threshold = drift_share_threshold
        self.critical_features = list(critical_features or ["Error_rate", "Latency_15min_avg", "Memory_growth_rate"])

    def evaluate(
        self,
        data_drift: DataDriftSummary,
        feature_metrics: dict[str, FeatureDriftResult],
        performance: PerformanceReport,
    ) -> RetrainingTriggerResult:
        reasons: list[str] = []
        recommendations: list[str] = []
        severity = "normal"

        # 1. Dataset-level feature drift check
        if data_drift.dataset_drift:
            pct = round(data_drift.drift_share * 100, 1)
            reasons.append(
                f"Dataset drift threshold breached: {data_drift.drifted_features_count}/{data_drift.total_features} "
                f"features ({pct}%) exhibit significant drift (threshold: {int(self.drift_share_threshold * 100)}%)."
            )
            severity = "warning"

        # 2. Critical individual feature drift
        crit_drifted = [f for f in self.critical_features if f in feature_metrics and feature_metrics[f].drift_level == "significant"]
        if crit_drifted:
            reasons.append(f"Critical operational feature(s) drifted significantly: {crit_drifted}.")
            severity = "warning"

        # 3. Model performance degradation check
        if performance.degraded:
            if performance.f1_drop is not None:
                reasons.append(
                    f"Model performance degradation: rolling F1 ({performance.rolling_metrics.f1:.4f}) dropped by "
                    f"{performance.f1_drop:.4f} compared to baseline ({performance.baseline_f1:.4f})."
                )
            else:
                reasons.append(
                    f"Model performance below acceptable threshold: rolling F1 is {performance.rolling_metrics.f1:.4f}."
                )
            severity = "critical"

        # Retraining required if either dataset drift or performance degradation occurred
        retraining_required = len(reasons) > 0

        if retraining_required:
            if severity == "critical":
                recommendations.append("Immediate model retraining required: model accuracy has degraded on recent production telemetry.")
            else:
                recommendations.append("Scheduled model retraining recommended: input feature distributions have shifted significantly from baseline.")
            if crit_drifted:
                recommendations.append(f"Inspect upstream service telemetry anomalies affecting: {crit_drifted}.")
            recommendations.append("Update feature baseline and evaluate retrained model candidates via experiment registry.")
        else:
            recommendations.append("System healthy: feature distributions and model performance are within nominal operating ranges.")

        return RetrainingTriggerResult(
            retraining_required=retraining_required,
            trigger_reasons=reasons,
            severity=severity,
            recommendations=recommendations,
        )


class DriftMonitoringEngine:
    """Orchestrates end-to-end drift detection, performance monitoring, and retraining checks."""

    def __init__(
        self,
        settings: dict[str, Any] | None = None,
        base_dir: str | Path | None = None,
    ) -> None:
        self.settings = settings or {}
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()

        drift_cfg = self.settings.get("mlops", {}).get("drift", {})
        psi_cfg = drift_cfg.get("psi", {})
        ks_cfg = drift_cfg.get("ks", {})
        perf_cfg = drift_cfg.get("performance", {})

        num_bins = int(psi_cfg.get("bins", 10))
        moderate_psi = float(psi_cfg.get("moderate_threshold", 0.10))
        significant_psi = float(psi_cfg.get("significant_threshold", 0.25))
        ks_alpha = float(ks_cfg.get("alpha", 0.05))
        drift_share_th = float(drift_cfg.get("dataset_drift_share_threshold", 0.30))

        self.report_path_rel = drift_cfg.get("report_path", "reports/evaluations/drift_monitoring_report.json")

        self.drift_detector = DriftDetector(
            num_bins=num_bins,
            moderate_psi=moderate_psi,
            significant_psi=significant_psi,
            ks_alpha=ks_alpha,
            dataset_drift_share_threshold=drift_share_th,
        )

        self.performance_monitor = PerformanceMonitor(
            f1_drop_threshold=float(perf_cfg.get("f1_drop_threshold", 0.15)),
            min_f1_threshold=float(perf_cfg.get("min_f1_threshold", 0.60)),
            min_samples=int(perf_cfg.get("min_samples", 50)),
        )

        self.retraining_trigger = RetrainingTrigger(
            drift_share_threshold=drift_share_th,
        )

    def run(
        self,
        reference_df: pd.DataFrame,
        target_df: pd.DataFrame,
        feature_columns: list[str] | None = None,
        y_true: Sequence[int] | np.ndarray | pd.Series | None = None,
        y_pred: Sequence[int] | np.ndarray | pd.Series | None = None,
        baseline_f1: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DriftMonitoringReport:
        # 1. Feature & Dataset drift
        summary, feature_metrics = self.drift_detector.analyze(
            reference_df, target_df, feature_columns=feature_columns
        )

        # 2. Performance monitoring
        if y_true is not None and y_pred is not None:
            perf_report = self.performance_monitor.evaluate(y_true, y_pred, baseline_f1=baseline_f1)
        else:
            perf_report = PerformanceReport(
                baseline_f1=baseline_f1,
                rolling_metrics=None,
                f1_drop=None,
                degraded=False,
                status="insufficient_data",
            )

        # 3. Retraining trigger evaluation
        trigger_res = self.retraining_trigger.evaluate(summary, feature_metrics, perf_report)

        report = DriftMonitoringReport(
            timestamp=utcnow(),
            reference_samples=len(reference_df),
            target_samples=len(target_df),
            data_drift=summary,
            feature_metrics=feature_metrics,
            performance=perf_report,
            trigger=trigger_res,
            metadata=metadata or {},
        )

        return report

    def save_report(self, report: DriftMonitoringReport, path: str | Path | None = None) -> Path:
        target_path = Path(path) if path else (self.base_dir / self.report_path_rel)
        report.to_json(target_path)
        return target_path
