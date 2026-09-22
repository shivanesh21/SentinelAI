from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .base import BaseDetector
from .ensemble import EnsembleDetector
from .metrics import binary_metrics, ranking_metrics
from .severity import DEFAULT_LEVELS, SeverityThresholds, calibrate_severity, level_distribution


def _resolve_class(class_name: str):
    if class_name in ("ZScoreDetector", "IQRTukeyDetector", "RollingIQRDetector"):
        from . import statistical as module
    elif class_name == "IsolationForestDetector":
        from . import isolation_forest as module
    elif class_name in ("DenseAutoencoderDetector", "LSTMAutoencoderDetector"):
        from . import autoencoder as module
    else:
        raise ValueError(f"unknown detector class: {class_name}")
    return getattr(module, class_name)


def build_detectors(settings: dict, interval_sec: int, include_lstm: bool = False) -> dict[str, BaseDetector]:
    anomaly = settings.get("anomaly", {}) or {}
    contamination = float(anomaly.get("contamination", 0.05))
    forest = anomaly.get("isolation_forest", {}) or {}
    ae = anomaly.get("autoencoder", {}) or {}
    ae_kwargs = dict(
        contamination=contamination,
        hidden_dims=tuple(ae.get("hidden_dims", [32, 8])),
        epochs=int(ae.get("epochs", 50)),
        batch_size=int(ae.get("batch_size", 256)),
        learning_rate=float(ae.get("learning_rate", 1e-3)),
        seed=int(ae.get("seed", 42)),
        normal_only=bool(ae.get("normal_only", True)),
    )
    detectors: dict[str, BaseDetector] = {
        "zscore": _resolve_class("ZScoreDetector")(contamination=contamination),
        "iqr": _resolve_class("IQRTukeyDetector")(contamination=contamination),
        "rolling_iqr": _resolve_class("RollingIQRDetector")(
            contamination=contamination,
            iqr_window_min=float(anomaly.get("rolling_iqr_window_min", 60)),
            interval_sec=interval_sec,
        ),
        "isolation_forest": _resolve_class("IsolationForestDetector")(
            contamination=contamination,
            n_estimators=int(forest.get("n_estimators", 200)),
            random_state=int(forest.get("random_state", 42)),
        ),
        "autoencoder": _resolve_class("DenseAutoencoderDetector")(**ae_kwargs),
    }
    if include_lstm or bool(ae.get("lstm_enabled", False)):
        detectors["lstm_autoencoder"] = _resolve_class("LSTMAutoencoderDetector")(
            seq_len=int(ae.get("seq_len", 20)), lstm_units=int(ae.get("lstm_units", 32)), **ae_kwargs
        )
    ensemble_cfg = anomaly.get("ensemble", {}) or {}
    members = [m for m in ensemble_cfg.get("members", ["zscore", "isolation_forest", "autoencoder"]) if m in detectors]
    detectors["ensemble"] = EnsembleDetector(
        member_names=members,
        strategy=ensemble_cfg.get("strategy", "mean_z"),
        weights=ensemble_cfg.get("weights", {}),
        contamination=contamination,
    )
    return detectors


class AnomalyDetectionModule:
    """Finalized anomaly module: fits a detector suite + score fusion, assigns severity
    buckets, evaluates against injected failure windows, and persists to disk."""

    def __init__(
        self,
        detectors: dict[str, BaseDetector],
        severity: SeverityThresholds | None = None,
        severities: dict[str, SeverityThresholds] | None = None,
        feature_columns: list[str] | None = None,
        group_col: str = "service",
        label_col: str = "failure_in_next_10min",
        primary: str = "ensemble",
    ):
        self.detectors = dict(detectors)
        self.severity = severity
        self.severities = dict(severities or {})
        self.feature_columns = list(feature_columns or [])
        self.group_col = group_col
        self.label_col = label_col
        self.primary = primary

    def fit(
        self,
        train: pd.DataFrame,
        feature_columns: list[str] | None = None,
        group_col: str | None = None,
        severity_quantiles: dict | None = None,
        levels: tuple[str, ...] = DEFAULT_LEVELS,
    ) -> "AnomalyDetectionModule":
        self.feature_columns = list(feature_columns or self.feature_columns)
        self.group_col = group_col or self.group_col

        for name, detector in self.detectors.items():
            if isinstance(detector, EnsembleDetector):
                continue
            detector.fit(train, self.feature_columns, self.group_col)
            detector.fit_threshold(detector.score(train))

        for detector in self.detectors.values():
            if isinstance(detector, EnsembleDetector):
                detector.set_members(self.detectors)
                detector.fit(train, self.feature_columns, self.group_col, fit_members=False)
                detector.fit_threshold(detector.score(train))

        self.severities = {
            name: calibrate_severity(detector.score(train), quantiles=severity_quantiles, levels=levels)
            for name, detector in self.detectors.items()
        }
        self.severity = self.severities.get(self.primary)
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[["timestamp", "service"]].copy() if {"timestamp", "service"}.issubset(df.columns) else pd.DataFrame(index=df.index)
        for name, detector in self.detectors.items():
            scores = detector.score(df)
            out[f"{name}_score"] = scores
            severity = self.severities.get(name)
            if severity is not None:
                out[f"{name}_level"] = severity.bucket(scores)
        if self.severity is not None and self.primary in self.detectors:
            out["anomaly_level"] = self.severity.bucket(self.detectors[self.primary].score(df))
            out["anomaly"] = (out["anomaly_level"] != DEFAULT_LEVELS[0]).astype(int)
        return out

    def evaluate(self, df: pd.DataFrame, split: str = "val") -> dict:
        y_next = df[self.label_col].to_numpy()
        y_fail = df["in_failure"].to_numpy() if "in_failure" in df.columns else None
        ttf = df["time_to_failure_min"].to_numpy() if "time_to_failure_min" in df.columns else None

        models: dict[str, dict] = {}
        for name, detector in self.detectors.items():
            scores = detector.score(df)
            severity = self.severities.get(name)
            levels = severity.bucket(scores) if severity is not None else np.full(len(df), DEFAULT_LEVELS[0], dtype=object)
            flagged = (levels != DEFAULT_LEVELS[0]).astype(int)

            entry: dict = {
                "threshold": round(float(detector.threshold_), 6) if detector.threshold_ is not None else None,
                "severity": severity.to_dict() if severity is not None else None,
                "levels": level_distribution(levels, severity.levels if severity is not None else DEFAULT_LEVELS),
                "detection": binary_metrics(y_next, flagged),
                "ranking": ranking_metrics(y_next, scores),
            }
            if y_fail is not None:
                entry["in_failure_detection"] = binary_metrics(y_fail, flagged)
            if ttf is not None:
                warning = flagged & (y_next == 1)
                lead = ttf[warning]
                entry["mean_lead_time_min"] = round(float(np.nanmean(lead)), 2) if len(lead) else None
            models[name] = entry

        return {
            "split": split,
            "rows": int(len(df)),
            "positive_rate_next": round(float(y_next.mean()), 4),
            "models": models,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        detector_info = {}
        for name, detector in self.detectors.items():
            if isinstance(detector, EnsembleDetector):
                detector.save(path / name)
                detector_info[name] = {"class": type(detector).__name__, "persist": "meta"}
            elif getattr(detector, "persist", "joblib") == "directory":
                detector.save(path / name)
                detector_info[name] = {"class": type(detector).__name__, "persist": "directory"}
            else:
                joblib.dump(detector, path / f"{name}.joblib")
                detector_info[name] = {"class": type(detector).__name__, "persist": "joblib"}

        meta = {
            "feature_columns": self.feature_columns,
            "group_col": self.group_col,
            "label_col": self.label_col,
            "primary": self.primary,
            "detectors": detector_info,
            "severities": {name: severity.to_dict() for name, severity in self.severities.items()},
        }
        (path / "module.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "AnomalyDetectionModule":
        path = Path(path)
        meta = json.loads((path / "module.json").read_text(encoding="utf-8"))
        detectors: dict[str, BaseDetector] = {}
        for name, info in meta["detectors"].items():
            persist = info["persist"]
            if persist == "meta":
                detectors[name] = EnsembleDetector.load(path / name)
            elif persist == "directory":
                detectors[name] = _resolve_class(info["class"]).load(path / name)
            else:
                detectors[name] = joblib.load(path / f"{name}.joblib")

        for detector in detectors.values():
            if isinstance(detector, EnsembleDetector):
                detector.set_members(detectors)

        module = cls(
            detectors=detectors,
            feature_columns=meta["feature_columns"],
            group_col=meta["group_col"],
            label_col=meta["label_col"],
            primary=meta["primary"],
        )
        module.severities = {name: SeverityThresholds.from_dict(data) for name, data in meta["severities"].items()}
        module.severity = module.severities.get(module.primary)
        return module
