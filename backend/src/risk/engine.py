from __future__ import annotations

import json
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

from src.models import (
    SequentialLSTMClassifier,
    build_sequences,
    load_advanced_model,
    load_baseline_model,
)

LEVELS = ("NORMAL", "WARNING", "ANOMALOUS", "CRITICAL")

MC = "lstm_classifier"


def level_from_score(score: float, thresholds: dict[str, float]) -> str:
    thresholds = dict(thresholds)
    if score >= float(thresholds.get("CRITICAL", 0.85)):
        return "CRITICAL"
    if score >= float(thresholds.get("ANOMALOUS", 0.65)):
        return "ANOMALOUS"
    if score >= float(thresholds.get("WARNING", 0.40)):
        return "WARNING"
    return "NORMAL"


def format_risk(score: float, thresholds: dict[str, float]) -> str:
    """Return formatted string like 'INCIDENT RISK 91% — CRITICAL'."""
    level = level_from_score(score, thresholds)
    pct = int(round(score * 100))
    return f"INCIDENT RISK {pct}% — {level}"


def _percentile_scores(values: np.ndarray, reference_sorted: np.ndarray) -> np.ndarray:
    if reference_sorted.size == 0:
        return np.full(len(values), np.nan, dtype=float)
    return np.searchsorted(reference_sorted, values, side="right") / reference_sorted.size


class RiskEngine:
    """Module 5 risk engine: fuse model probabilities + operational signals into a
    per-tick risk score with severity levels.

    Components (weights / thresholds from settings.risk):
      - failure_probability: mean of the loaded classifiers' predicted probabilities
        (baseline / advanced / LSTM, the LSTM only where a window is available).
      - error_rate / latency_trend / memory_growth: percentile-ranks of the mapped
        features relative to the training reference, bounded to [0, 1].
      - anomaly_score: reserved for an externally-fitted anomaly detector (added at
        construction if an object with ``score(df)`` is provided).
    """

    _MODEL_DIRS = {
        "logistic_regression": ("baseline", "logistic_regression", "baseline"),
        "random_forest": ("baseline", "random_forest", "baseline"),
        "xgboost": ("advanced", "xgboost", "advanced"),
        "lightgbm": ("advanced", "lightgbm", "advanced"),
    }

    def __init__(
        self,
        models: dict[str, object] | None = None,
        feature_columns: list[str] | None = None,
        weights: dict[str, float] | None = None,
        severity_thresholds: dict[str, float] | None = None,
        signal_map: dict[str, str] | None = None,
        anomaly_detector: object | None = None,
        label_col: str = "failure_in_next_10min",
        group_col: str = "service",
    ):
        self.models = OrderedDict(models or {})
        self.feature_columns = list(feature_columns or [])
        self.weights = dict(weights or {})
        self.severity_thresholds = dict(severity_thresholds or {})
        self.signal_map = dict(signal_map or {})
        self.anomaly_detector = anomaly_detector
        self.label_col = label_col
        self.group_col = group_col
        self.references: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(cls, models_dir: str | Path, settings: dict | None = None) -> "RiskEngine":
        models_dir = Path(models_dir)
        settings = settings or {}
        risk = settings.get("risk", {}) or {}

        models: OrderedDict[str, object] = OrderedDict()
        feature_columns: list[str] | None = None
        for name, (folder, subdir, kind) in cls._MODEL_DIRS.items():
            path = models_dir / folder / subdir
            if not path.exists():
                continue
            try:
                if kind == "baseline":
                    model, _, features, _ = load_baseline_model(path)
                else:
                    model, _, features, _ = load_advanced_model(path)
            except Exception:
                continue
            models[name] = model
            feature_columns = features or feature_columns

        lstm_path = models_dir / "prediction" / MC
        if lstm_path.exists() and SequentialLSTMClassifier is not None:
            models[MC] = SequentialLSTMClassifier.load(lstm_path)
            feature_columns = feature_columns or None

        if not feature_columns:
            raise ValueError("no persisted models found; train the models first")

        engine = cls(
            models=models,
            feature_columns=feature_columns,
            weights=risk.get("weights", {}),
            severity_thresholds=risk.get("severity_thresholds", {}),
            signal_map=risk.get("signal_map", {}),
        )
        calibration_path = models_dir / "risk" / "calibration.json"
        if calibration_path.exists():
            engine._load_calibration(calibration_path)
        return engine

    # --------------------------------------------------------- training support

    def calibrate(self, train: pd.DataFrame) -> "RiskEngine":
        for component, feature in self.signal_map.items():
            if feature in train.columns:
                values = train[feature].to_numpy(dtype=float)
                values = values[~np.isnan(values)]
                self.references[component] = np.sort(values)
        return self

    def available_components(self) -> list[str]:
        components = list(self.weights)
        if MC not in self.models:
            components = [c for c in components if c != "failure_probability"]
            components.insert(0, "failure_probability")
        if self.anomaly_detector is None and "anomaly_score" in components:
            components.remove("anomaly_score")
        return [c for c in components if c in self.weights]

    def _component_weights(self) -> dict[str, float]:
        available = self.available_components()
        total = sum(float(self.weights.get(c, 0.0)) for c in available) or 1.0
        return {c: float(self.weights.get(c, 0.0)) / total for c in available}

    # ------------------------------------------------------------------ scoring

    def _probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        probs = {}
        for name, model in self.models.items():
            if name == MC:
                continue
            required = getattr(model, "feature_columns_in", None) or self.feature_columns
            try:
                p = model.predict_proba(df[required].to_numpy(dtype=float))[:, 1]
            except KeyError:
                p = np.full(len(df), np.nan, dtype=float)
            probs[name] = p

        if MC in self.models and len(df):
            lstm = self.models[MC]
            working = df.copy()
            if self.label_col not in working.columns:
                working[self.label_col] = 0
            X, _, meta = build_sequences(
                working,
                self.feature_columns,
                self.label_col,
                seq_len=lstm.config.seq_len,
                group_col=self.group_col,
            )
            lstm_prob = np.full(len(df), np.nan, dtype=float)
            if len(X):
                window_prob = lstm.predict_proba(X)
                for (ts, service), p in zip(meta[["timestamp", self.group_col]].itertuples(index=False), window_prob):
                    mask = (df[self.group_col] == service) & (df["timestamp"] == ts)
                    if mask.any():
                        idx = np.flatnonzero(mask.to_numpy())[0]
                        lstm_prob[idx] = float(p)
            probs[MC] = lstm_prob

        out = pd.DataFrame(probs, index=df.index)
        out["failure_probability"] = out[list(self.models)].mean(axis=1, skipna=True)
        return out

    def components(self, df: pd.DataFrame) -> pd.DataFrame:
        if not {"timestamp", self.group_col}.issubset(df.columns):
            raise ValueError("score requires timestamp and service columns")
        out = df[["timestamp", self.group_col]].copy()
        out = pd.concat([out, self._probabilities(df)], axis=1)

        for component, feature in self.signal_map.items():
            ref = self.references.get(component)
            if ref is None:
                continue
            raw = df[feature].to_numpy(dtype=float)
            out[component] = _percentile_scores(raw, ref)

        if self.anomaly_detector is not None:
            anomaly = self.anomaly_detector.score(df)
            ref = np.sort(np.asarray(anomaly[f"{self.anomaly_detector.primary}_score"].to_numpy(dtype=float)))
            out["anomaly_score"] = _percentile_scores(
                anomaly[f"{self.anomaly_detector.primary}_score"].to_numpy(dtype=float), ref
            )
        return out

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        comps = self.components(df)
        weights = self._component_weights()
        components = list(weights)
        values = comps[components].to_numpy(dtype=float)
        values = np.nan_to_num(values, nan=0.0)
        weighted = np.asarray(list(weights.values()))[None, :]
        mask = np.isfinite(comps[components].to_numpy(dtype=float))
        weights_matrix = np.where(mask, weighted, 0.0)
        row_sum = weights_matrix.sum(axis=1)
        row_sum = np.where(row_sum == 0, 1.0, row_sum)
        risk_score = (weights_matrix * values).sum(axis=1) / row_sum

        comps["risk_score"] = np.clip(risk_score, 0.0, 1.0)
        comps["risk_level"] = [
            level_from_score(float(s), self.severity_thresholds) for s in comps["risk_score"].to_numpy()
        ]
        comps["risk"] = (comps["risk_level"] != LEVELS[0]).astype(int)
        return comps

    # ------------------------------------------------------------ aggregation

    def latest_by_service(self, df: pd.DataFrame) -> pd.DataFrame:
        scored = self.score(df)
        scored = scored.sort_values("timestamp")
        latest = scored.drop_duplicates(subset=[self.group_col], keep="last").copy()
        latest = latest.sort_values("risk_score", ascending=False)
        return latest.reset_index(drop=True)

    # ------------------------------------------------------------- persistence

    def save_calibration(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_columns": self.feature_columns,
            "weights": self.weights,
            "severity_thresholds": self.severity_thresholds,
            "signal_map": self.signal_map,
            "references": {k: v.tolist() for k, v in self.references.items()},
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _load_calibration(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.feature_columns = list(payload.get("feature_columns", self.feature_columns))
        self.weights = payload.get("weights", self.weights)
        self.severity_thresholds = payload.get("severity_thresholds", self.severity_thresholds)
        self.signal_map = payload.get("signal_map", self.signal_map)
        self.references = {
            k: np.asarray(v, dtype=float) for k, v in payload.get("references", {}).items()
        }