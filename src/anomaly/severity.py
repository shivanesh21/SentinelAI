from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_LEVELS = ("NORMAL", "WARNING", "ANOMALOUS", "CRITICAL")
DEFAULT_QUANTILES = {"warning": 0.95, "anomalous": 0.975, "critical": 0.99}


@dataclass
class SeverityThresholds:
    warning: float
    anomalous: float
    critical: float
    levels: tuple[str, ...] = DEFAULT_LEVELS

    def bucket(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        labels = np.full(len(scores), self.levels[0], dtype=object)
        labels[scores > self.warning] = self.levels[1]
        labels[scores > self.anomalous] = self.levels[2]
        labels[scores > self.critical] = self.levels[3]
        return labels

    def to_dict(self) -> dict:
        return {
            "warning": float(self.warning),
            "anomalous": float(self.anomalous),
            "critical": float(self.critical),
            "levels": list(self.levels),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SeverityThresholds":
        return cls(
            warning=float(data["warning"]),
            anomalous=float(data["anomalous"]),
            critical=float(data["critical"]),
            levels=tuple(data.get("levels", DEFAULT_LEVELS)),
        )


def calibrate_severity(
    train_scores: np.ndarray,
    quantiles: dict | None = None,
    levels: tuple[str, ...] = DEFAULT_LEVELS,
) -> SeverityThresholds:
    # Boundaries are quantiles of the normal training score distribution; higher
    # boundary -> smaller bucket. Bumps are forced strictly increasing so scores
    # never fall into an impossible bucket.
    q = {**DEFAULT_QUANTILES, **(quantiles or {})}
    warning = float(np.quantile(train_scores, q["warning"]))
    anomalous = float(np.quantile(train_scores, q["anomalous"]))
    critical = float(np.quantile(train_scores, q["critical"]))
    anomalous = max(anomalous, warning + 1e-12)
    critical = max(critical, anomalous + 1e-12)
    return SeverityThresholds(warning=warning, anomalous=anomalous, critical=critical, levels=levels)


def level_distribution(labels: np.ndarray, levels: tuple[str, ...] = DEFAULT_LEVELS) -> dict:
    total = len(labels)
    return {
        level: {
            "count": int((labels == level).sum()),
            "share": round(float((labels == level).mean()), 4) if total else 0.0,
        }
        for level in levels
    }
