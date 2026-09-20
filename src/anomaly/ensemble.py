from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseDetector


class EnsembleDetector(BaseDetector):
    """Fuse member detector scores into a single anomaly score.

    ``mean_z`` standardises each member's score using its training mean/std and averages
    them; ``mean_rank`` converts each member score to its percentile in the training
    distribution before averaging. Both make members with different scales comparable.
    Members must be fitted before (or by) ``fit``.
    """

    name = "ensemble"
    persist = "meta"

    def __init__(
        self,
        member_names: list[str],
        strategy: str = "mean_z",
        weights: dict | None = None,
        contamination: float = 0.05,
    ):
        super().__init__(contamination=contamination)
        if strategy not in ("mean_z", "mean_rank"):
            raise ValueError(f"unknown ensemble strategy: {strategy}")
        self.member_names = list(member_names)
        self.strategy = strategy
        self.weights = dict(weights or {})
        self.members_: dict[str, BaseDetector] = {}
        self.calibration_: dict = {}

    def set_members(self, members: dict[str, BaseDetector]) -> "EnsembleDetector":
        missing = [name for name in self.member_names if name not in members]
        if missing:
            raise ValueError(f"missing ensemble members: {missing}")
        self.members_ = {name: members[name] for name in self.member_names}
        return self

    def fit(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        group_col: str = "service",
        fit_members: bool = True,
    ) -> "EnsembleDetector":
        self.feature_columns = list(feature_columns)
        self.group_col = group_col
        if fit_members:
            for member in self.members_.values():
                member.fit(df, feature_columns, group_col)
        raw = self._member_scores(df)
        self._calibrate(raw)
        return self

    def _member_scores(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        if not self.members_:
            raise ValueError("ensemble has no members; call set_members() first")
        return {name: np.asarray(member.score(df), dtype=float) for name, member in self.members_.items()}

    def _calibrate(self, raw: dict[str, np.ndarray]) -> None:
        self.calibration_ = {}
        for name, scores in raw.items():
            if self.strategy == "mean_rank":
                self.calibration_[name] = np.sort(scores)
            else:
                std = float(np.std(scores)) or 1.0
                self.calibration_[name] = (float(np.mean(scores)), std)

    def _normalize(self, name: str, scores: np.ndarray) -> np.ndarray:
        reference = self.calibration_.get(name)
        if reference is None:
            return np.asarray(scores, dtype=float)
        if self.strategy == "mean_rank":
            return np.searchsorted(reference, scores, side="right") / max(len(reference), 1)
        mean, std = reference
        return (np.asarray(scores, dtype=float) - mean) / std

    def _score_impl(self, df: pd.DataFrame) -> pd.Series:
        raw = self._member_scores(df)
        combined = np.zeros(len(df), dtype=float)
        total_weight = 0.0
        for name, scores in raw.items():
            weight = float(self.weights.get(name, 1.0))
            combined += weight * self._normalize(name, scores)
            total_weight += weight
        if total_weight:
            combined /= total_weight
        return pd.Series(combined, index=df.index)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "member_names": self.member_names,
            "strategy": self.strategy,
            "weights": self.weights,
            "contamination": self.contamination,
            "threshold": self.threshold_,
            "feature_columns": self.feature_columns,
            "group_col": self.group_col,
            "calibration": self.calibration_,
        }
        with (path / "ensemble.pkl").open("wb") as handle:
            pickle.dump(meta, handle)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "EnsembleDetector":
        with (Path(path) / "ensemble.pkl").open("rb") as handle:
            meta = pickle.load(handle)
        detector = cls(
            member_names=meta["member_names"],
            strategy=meta["strategy"],
            weights=meta["weights"],
            contamination=meta["contamination"],
        )
        detector.threshold_ = meta["threshold"]
        detector.feature_columns = meta["feature_columns"]
        detector.group_col = meta["group_col"]
        detector.calibration_ = meta["calibration"]
        return detector
