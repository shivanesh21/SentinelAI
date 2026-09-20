from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseDetector, iter_groups


class ZScoreDetector(BaseDetector):
    """Per-feature z-score baseline; score = max |z| across features."""

    name = "zscore"

    def fit(self, df: pd.DataFrame, feature_columns: list[str], group_col: str = "service") -> "ZScoreDetector":
        self.feature_columns = list(feature_columns)
        self.group_col = group_col
        self.stats_: dict = {}
        for key, sub in iter_groups(df, group_col):
            matrix = sub[self.feature_columns].to_numpy(dtype=float)
            mean = matrix.mean(axis=0)
            std = np.maximum(matrix.std(axis=0), self.eps)
            self.stats_[key if key is not None else "__all__"] = (mean, std)
        return self

    def _score_impl(self, df: pd.DataFrame) -> pd.Series:
        out = pd.Series(0.0, index=df.index)
        fallback = self.stats_.get("__all__")
        for key, sub in iter_groups(df, self.group_col):
            mean, std = self.stats_.get(key if key is not None else "__all__", fallback)
            matrix = sub[self.feature_columns].to_numpy(dtype=float)
            z = np.abs((matrix - mean) / std)
            out.loc[sub.index] = z.max(axis=1)
        return out

    def train_scores(self, df: pd.DataFrame) -> np.ndarray:
        return self.score(df)


class IQRTukeyDetector(BaseDetector):
    """Tukey IQR-fence baseline; score = max normalized excursion beyond Q1/Q3."""

    name = "iqr"

    def fit(self, df: pd.DataFrame, feature_columns: list[str], group_col: str = "service") -> "IQRTukeyDetector":
        self.feature_columns = list(feature_columns)
        self.group_col = group_col
        self.stats_: dict = {}
        for key, sub in iter_groups(df, group_col):
            matrix = sub[self.feature_columns].to_numpy(dtype=float)
            q1, q3 = np.quantile(matrix, 0.25, axis=0), np.quantile(matrix, 0.75, axis=0)
            iqr = np.maximum(q3 - q1, self.eps)
            self.stats_[key if key is not None else "__all__"] = (q1, q3, iqr)
        return self

    def _score_impl(self, df: pd.DataFrame) -> pd.Series:
        out = pd.Series(0.0, index=df.index)
        fallback = self.stats_.get("__all__")
        for key, sub in iter_groups(df, self.group_col):
            q1, q3, iqr = self.stats_.get(key if key is not None else "__all__", fallback)
            matrix = sub[self.feature_columns].to_numpy(dtype=float)
            excursion = np.maximum(np.maximum(q1 - matrix, matrix - q3), 0.0)
            out.loc[sub.index] = (excursion / iqr).max(axis=1)
        return out


class RollingIQRDetector(BaseDetector):
    """Time-aware rolling IQR baseline computed per service.

    ``iqr_window_min`` minutes of rolling history are used to derive the median and
    inter-quartile fence, and the score is the current value's normalized excursion
    beyond those local fences. Requires ``timestamp`` (and ideally ``service``) columns.
    """

    name = "rolling_iqr"

    def __init__(
        self,
        contamination: float = 0.05,
        iqr_window_min: float = 60.0,
        interval_sec: int = 30,
        eps: float = 1e-9,
    ):
        super().__init__(contamination=contamination, eps=eps)
        self.iqr_window_min = float(iqr_window_min)
        self.interval_sec = int(interval_sec)

    @property
    def window_ticks(self) -> int:
        return max(2, int(round(self.iqr_window_min * 60.0 / self.interval_sec)))

    def fit(self, df: pd.DataFrame, feature_columns: list[str], group_col: str = "service") -> "RollingIQRDetector":
        self.feature_columns = list(feature_columns)
        self.group_col = group_col
        return self

    def _score_impl(self, df: pd.DataFrame) -> pd.Series:
        out = pd.Series(0.0, index=df.index)
        window = self.window_ticks
        for _, sub in iter_groups(df, self.group_col):
            ordered = sub.sort_values("timestamp") if "timestamp" in sub.columns else sub
            features = ordered[self.feature_columns]
            median = features.rolling(window, min_periods=1).median()
            q1 = features.rolling(window, min_periods=1).quantile(0.25)
            q3 = features.rolling(window, min_periods=1).quantile(0.75)
            iqr = (q3 - q1).clip(lower=self.eps)
            excursion = np.maximum(np.maximum(q1 - features, features - q3), 0.0)
            out.loc[ordered.index] = (excursion / iqr).max(axis=1)
        return out
