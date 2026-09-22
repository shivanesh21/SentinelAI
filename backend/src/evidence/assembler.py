from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.risk.engine import RiskEngine


@dataclass
class Evidence:
    timestamp: str
    service: str
    risk_score: float
    risk_level: str
    failure_probability: float
    components: dict[str, float]
    top_anomalous_features: list[dict[str, Any]]
    metrics_deltas: dict[str, float]
    correlated_logs: list[dict[str, Any]]
    recommended_actions: list[str]


class EvidenceAssembler:
    """Module 6: packages structured evidence when risk crosses threshold."""

    def __init__(
        self,
        engine: RiskEngine,
        threshold: float = 0.40,  # WARNING level
        feature_columns: list[str] | None = None,
        top_k_features: int = 5,
    ):
        self.engine = engine
        self.threshold = threshold
        self.feature_columns = feature_columns or []
        self.top_k_features = top_k_features

    def assemble(self, df: pd.DataFrame) -> list[Evidence]:
        """Assemble evidence for all rows where risk_score >= threshold."""
        scored = self.engine.score(df)
        flagged = scored[scored["risk_score"] >= self.threshold].copy()

        if flagged.empty:
            return []

        evidence_list = []
        for _, row in flagged.iterrows():
            evidence = self._build_evidence(row, df)
            evidence_list.append(evidence)

        return evidence_list

    def _build_evidence(self, row: pd.Series, full_df: pd.DataFrame) -> Evidence:
        service = row["service"]
        timestamp = row["timestamp"]

        # Service-specific history (last 30 min)
        svc_df = full_df[full_df["service"] == service].sort_values("timestamp")
        recent = svc_df[svc_df["timestamp"] <= timestamp].tail(60)  # 30 min at 30s intervals

        # Components dict
        components = {
            "failure_probability": float(row.get("failure_probability", 0.0)),
            "error_rate": float(row.get("error_rate", 0.0)),
            "latency_trend": float(row.get("latency_trend", 0.0)),
            "memory_growth": float(row.get("memory_growth", 0.0)),
        }
        if "anomaly_score" in row:
            components["anomaly_score"] = float(row["anomaly_score"])

        # Top anomalous features (by deviation from service baseline)
        top_features = self._top_anomalous_features(recent, timestamp)

        # Metrics deltas (current vs 10-min ago baseline)
        metrics_deltas = self._metrics_deltas(recent, timestamp)

        # Correlated logs (simulated - in production would query log store)
        correlated_logs = self._simulate_correlated_logs(service, timestamp, row["risk_level"])

        # Recommended actions based on dominant signals
        recommended_actions = self._recommend_actions(components, top_features)

        return Evidence(
            timestamp=timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
            service=service,
            risk_score=float(row["risk_score"]),
            risk_level=row["risk_level"],
            failure_probability=float(row["failure_probability"]),
            components=components,
            top_anomalous_features=top_features,
            metrics_deltas=metrics_deltas,
            correlated_logs=correlated_logs,
            recommended_actions=recommended_actions,
        )

    def _top_anomalous_features(self, recent: pd.DataFrame, timestamp) -> list[dict[str, Any]]:
        """Find features most deviating from their recent baseline."""
        if len(recent) < 10:
            return []

        baseline = recent.iloc[:-1]  # exclude current
        current = recent.iloc[-1]

        deviations = []
        for feat in self.feature_columns:
            if feat not in recent.columns:
                continue
            baseline_vals = baseline[feat].to_numpy(dtype=float)
            baseline_vals = baseline_vals[~np.isnan(baseline_vals)]
            if len(baseline_vals) < 5:
                continue
            mean = float(np.mean(baseline_vals))
            std = float(np.std(baseline_vals)) + 1e-6
            current_val = float(current[feat])
            zscore = (current_val - mean) / std
            if abs(zscore) > 1.5:
                deviations.append({
                    "feature": feat,
                    "current": current_val,
                    "baseline_mean": mean,
                    "baseline_std": std,
                    "zscore": round(zscore, 2),
                    "direction": "up" if zscore > 0 else "down",
                })

        deviations.sort(key=lambda x: abs(x["zscore"]), reverse=True)
        return deviations[: self.top_k_features]

    def _metrics_deltas(self, recent: pd.DataFrame, timestamp) -> dict[str, float]:
        """Compute key metrics deltas vs 10-min ago."""
        if len(recent) < 20:
            return {}

        current = recent.iloc[-1]
        baseline = recent.iloc[-20]  # ~10 min ago at 30s intervals

        deltas = {}
        key_metrics = [
            "Error_rate", "Latency_15min_avg", "Memory_growth_rate",
            "CPU_15min_avg", "Request_rate", "DB_usage_15min_avg"
        ]
        for metric in key_metrics:
            if metric in recent.columns:
                cur = float(current[metric])
                base = float(baseline[metric])
                if base != 0:
                    deltas[metric] = round((cur - base) / base * 100, 1)
                else:
                    deltas[metric] = round(cur - base, 1)
        return deltas

    def _simulate_correlated_logs(self, service: str, timestamp, risk_level: str) -> list[dict[str, Any]]:
        """Simulate correlated log entries (in production, query Loki/Elastic)."""
        log_templates = {
            "CRITICAL": [
                {"level": "ERROR", "message": f"[{service}] Connection pool exhausted - 0 available"},
                {"level": "ERROR", "message": f"[{service}] Request timeout after 30s - downstream unavailable"},
                {"level": "WARN", "message": f"[{service}] Memory usage > 90% - GC thrashing detected"},
            ],
            "ANOMALOUS": [
                {"level": "WARN", "message": f"[{service}] Elevated error rate: 5xx responses increased 3x"},
                {"level": "WARN", "message": f"[{service}] Latency p99 > 5s - queue buildup"},
                {"level": "INFO", "message": f"[{service}] Autoscaler triggered - adding 2 replicas"},
            ],
            "WARNING": [
                {"level": "WARN", "message": f"[{service}] Latency trending up - 20% increase in 10min"},
                {"level": "INFO", "message": f"[{service}] New deployment started - v2.1.4 rolling out"},
            ],
            "NORMAL": [
                {"level": "INFO", "message": f"[{service}] Health check OK - all systems nominal"},
            ],
        }
        return log_templates.get(risk_level, log_templates["NORMAL"])

    def _recommend_actions(self, components: dict, top_features: list[dict]) -> list[str]:
        """Generate recommended actions based on dominant signals."""
        actions = []

        if components.get("failure_probability", 0) > 0.7:
            actions.append("Escalate to on-call: high failure probability detected")

        if components.get("error_rate", 0) > 0.7:
            actions.append("Check downstream dependencies and circuit breakers")
            actions.append("Review recent deployments for regression")

        if components.get("latency_trend", 0) > 0.7:
            actions.append("Scale horizontally: add replicas to reduce queue depth")
            actions.append("Enable request shedding for non-critical paths")

        if components.get("memory_growth", 0) > 0.7:
            actions.append("Trigger heap dump and memory profile analysis")
            actions.append("Consider rolling restart to reclaim memory")

        # Feature-specific
        for feat in top_features[:3]:
            if "memory" in feat["feature"].lower() and feat["direction"] == "up":
                actions.append(f"Investigate {feat['feature']} growth - possible leak")
            if "latency" in feat["feature"].lower() and feat["direction"] == "up":
                actions.append(f"Profile {feat['feature']} - check for contention or slow queries")
            if "error" in feat["feature"].lower() and feat["direction"] == "up":
                actions.append(f"Analyze {feat['feature']} - correlate with upstream errors")

        return list(dict.fromkeys(actions))  # deduplicate preserving order


def assemble_evidence(
    engine: RiskEngine,
    df: pd.DataFrame,
    threshold: float = 0.40,
    feature_columns: list[str] | None = None,
) -> list[Evidence]:
    """Convenience function."""
    assembler = EvidenceAssembler(engine, threshold, feature_columns)
    return assembler.assemble(df)


def evidence_to_json(evidence: Evidence) -> str:
    """Serialize evidence to JSON."""
    return json.dumps(asdict(evidence), indent=2, default=str)