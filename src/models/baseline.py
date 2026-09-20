from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ModelName = Literal["logistic_regression", "random_forest"]


@dataclass
class BaselineModelConfig:
    name: ModelName
    params: dict | None = None

    def create(self) -> Pipeline:
        if self.name == "logistic_regression":
            params = {
                "penalty": "l2",
                "C": 1.0,
                "class_weight": "balanced",
                "max_iter": 1000,
                "solver": "lbfgs",
                "random_state": 42,
                **(self.params or {}),
            }
            clf = LogisticRegression(**params)
            return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        if self.name == "random_forest":
            params = {
                "n_estimators": 200,
                "max_depth": 12,
                "min_samples_split": 10,
                "min_samples_leaf": 4,
                "class_weight": "balanced_subsample",
                "random_state": 42,
                "n_jobs": -1,
                **(self.params or {}),
            }
            clf = RandomForestClassifier(**params)
            return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        raise ValueError(f"unknown baseline model: {self.name}")


@dataclass
class BaselineReport:
    model_name: str
    split: str
    threshold: float
    metrics: dict
    feature_importance: list[dict] | None = None

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "split": self.split,
            "threshold": self.threshold,
            "metrics": self.metrics,
            "feature_importance": self.feature_importance,
        }


def evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "positive_rate": float(y_true.mean()),
        "predicted_positive_rate": float(y_pred.mean()),
    }


def find_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray, metric: str = "f1") -> float:
    from sklearn.metrics import f1_score, precision_recall_curve

    if metric == "f1":
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
        idx = int(np.nanargmax(f1_scores))
        return float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return 0.5


def train_baseline(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_columns: list[str],
    label_col: str = "failure_in_next_10min",
    model_config: BaselineModelConfig | None = None,
) -> tuple[Pipeline, BaselineReport]:
    model_config = model_config or BaselineModelConfig(name="logistic_regression")
    pipe = model_config.create()

    X_train = train[feature_columns].to_numpy(dtype=float)
    y_train = train[label_col].to_numpy(dtype=int)
    pipe.fit(X_train, y_train)

    X_val = val[feature_columns].to_numpy(dtype=float)
    y_val = val[label_col].to_numpy(dtype=int)
    val_scores = pipe.predict_proba(X_val)[:, 1]
    threshold = find_optimal_threshold(y_val, val_scores)
    val_pred = (val_scores >= threshold).astype(int)

    metrics = evaluate_binary(y_val, val_pred, val_scores)

    fi = None
    if model_config.name == "random_forest":
        rf = pipe.named_steps["clf"]
        fi = [
            {"feature": feature_columns[i], "importance": float(rf.feature_importances_[i])}
            for i in np.argsort(rf.feature_importances_)[::-1]
        ]
    elif model_config.name == "logistic_regression":
        lr = pipe.named_steps["clf"]
        coef = lr.coef_[0]
        fi = [
            {"feature": feature_columns[i], "importance": float(coef[i])}
            for i in np.argsort(np.abs(coef))[::-1]
        ]

    report = BaselineReport(
        model_name=model_config.name,
        split="val",
        threshold=threshold,
        metrics=metrics,
        feature_importance=fi,
    )
    return pipe, report


def evaluate_split(pipe: Pipeline, df: pd.DataFrame, feature_columns: list[str], label_col: str, threshold: float) -> dict:
    X = df[feature_columns].to_numpy(dtype=float)
    y = df[label_col].to_numpy(dtype=int)
    scores = pipe.predict_proba(X)[:, 1]
    preds = (scores >= threshold).astype(int)
    return evaluate_binary(y, preds, scores)


def save_model(pipe: Pipeline, path: str | Path, threshold: float, feature_columns: list[str], meta: dict | None = None) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path / "model.joblib")
    with (path / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold": float(threshold),
                "feature_columns": feature_columns,
                "meta": meta or {},
            },
            f,
            indent=2,
        )
    return path


def load_model(path: str | Path) -> tuple[Pipeline, float, list[str], dict]:
    path = Path(path)
    pipe = joblib.load(path / "model.joblib")
    with (path / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return pipe, float(meta["threshold"]), meta["feature_columns"], meta.get("meta", {})


def build_model_configs(settings: dict) -> dict[str, BaselineModelConfig]:
    models = settings.get("models", {}).get("baseline", {}) or {}
    return {
        "logistic_regression": BaselineModelConfig(
            name="logistic_regression", params=models.get("logistic_regression", {})
        ),
        "random_forest": BaselineModelConfig(
            name="random_forest", params=models.get("random_forest", {})
        ),
    }