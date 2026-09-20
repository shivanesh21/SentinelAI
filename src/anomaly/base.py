from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class BaseDetector:
    """Common interface: fit on train, score rows (higher = more anomalous)."""

    name = "base"
    persist = "joblib"

    def __init__(self, contamination: float = 0.05, eps: float = 1e-9):
        self.contamination = float(contamination)
        self.eps = float(eps)
        self.feature_columns: list[str] = []
        self.group_col: str = "service"
        self.threshold_: float | None = None

    def fit(self, df: pd.DataFrame, feature_columns: list[str], group_col: str = "service") -> "BaseDetector":
        raise NotImplementedError

    def _score_impl(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._score_impl(df), dtype=float)

    def fit_threshold(self, scores: np.ndarray) -> float:
        self.threshold_ = float(np.quantile(scores, 1.0 - self.contamination))
        return self.threshold_

    def predict(self, df: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        scores = self.score(df)
        thr = self.threshold_ if threshold is None else threshold
        if thr is None:
            thr = float(np.quantile(scores, 1.0 - self.contamination))
        return (scores > thr).astype(int)

    def score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[["timestamp", "service"]].copy()
        out["score"] = self.score(df)
        thr = self.threshold_ if self.threshold_ is not None else float(np.quantile(out["score"], 1.0 - self.contamination))
        out["anomaly"] = (out["score"] > thr).astype(int)
        return out

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BaseDetector":
        return joblib.load(path)


def iter_groups(df: pd.DataFrame, group_col: str | None):
    if group_col and group_col in df.columns:
        yield from df.groupby(group_col, sort=False)
    else:
        yield None, df


def assign_levels(
    scores: np.ndarray,
    train_scores: np.ndarray,
    levels: tuple[str, ...] = ("NORMAL", "WARNING", "ANOMALOUS", "CRITICAL"),
    contamination: float = 0.05,
) -> np.ndarray:
    normal_thr = float(np.quantile(train_scores, 0.5))
    warning_thr = float(np.quantile(train_scores, 1.0 - contamination))
    critical_thr = float(np.quantile(train_scores, 1.0 - contamination / 2.0))
    labels = np.full(len(scores), levels[0], dtype=object)
    labels[scores > normal_thr] = levels[1]
    labels[scores > warning_thr] = levels[2]
    labels[scores > critical_thr] = levels[3]
    return labels
