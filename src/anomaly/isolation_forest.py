from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .base import BaseDetector, iter_groups


class IsolationForestDetector(BaseDetector):
    """Isolation Forest on engineered features, fitted per service."""

    name = "isolation_forest"

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        max_samples: str | int | float = "auto",
        random_state: int = 42,
    ):
        super().__init__(contamination=contamination)
        self.n_estimators = int(n_estimators)
        self.max_samples = max_samples
        self.random_state = int(random_state)
        self.models_: dict = {}
        self.global_model_: IsolationForest | None = None

    def _make(self) -> IsolationForest:
        return IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def fit(self, df: pd.DataFrame, feature_columns: list[str], group_col: str = "service") -> "IsolationForestDetector":
        self.feature_columns = list(feature_columns)
        self.group_col = group_col
        self.models_ = {}
        for key, sub in iter_groups(df, group_col):
            model = self._make()
            model.fit(sub[self.feature_columns].to_numpy(dtype=float))
            self.models_[key if key is not None else "__all__"] = model
        self.global_model_ = self._make()
        self.global_model_.fit(df[self.feature_columns].to_numpy(dtype=float))
        return self

    def _score_impl(self, df: pd.DataFrame) -> pd.Series:
        out = pd.Series(0.0, index=df.index)
        for key, sub in iter_groups(df, self.group_col):
            model = self.models_.get(key if key is not None else "__all__", self.global_model_)
            matrix = sub[self.feature_columns].to_numpy(dtype=float)
            out.loc[sub.index] = -model.score_samples(matrix)
        return out
