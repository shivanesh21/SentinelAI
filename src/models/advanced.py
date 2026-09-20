from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier


ModelName = Literal["xgboost", "lightgbm"]


@dataclass
class AdvancedModelConfig:
    name: ModelName
    params: dict | None = None
    tune: bool = False
    tune_trials: int = 30
    tune_cv: int = 3

    def create(self) -> XGBClassifier | LGBMClassifier:
        if self.name == "xgboost":
            params = {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 3,
                "reg_lambda": 1.0,
                "reg_alpha": 0.0,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "scale_pos_weight": 1.0,
                "random_state": 42,
                "n_jobs": -1,
                "tree_method": "hist",
                **(self.params or {}),
            }
            return XGBClassifier(**params)
        if self.name == "lightgbm":
            params = {
                "n_estimators": 300,
                "max_depth": -1,
                "num_leaves": 63,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_samples": 20,
                "reg_lambda": 1.0,
                "reg_alpha": 0.0,
                "objective": "binary",
                "metric": "auc",
                "scale_pos_weight": 1.0,
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": -1,
                **(self.params or {}),
            }
            return LGBMClassifier(**params)
        raise ValueError(f"unknown advanced model: {self.name}")

    def param_dist(self) -> dict:
        if self.name == "xgboost":
            return {
                "n_estimators": [200, 300, 400, 500],
                "max_depth": [4, 5, 6, 7, 8],
                "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                "min_child_weight": [1, 3, 5, 7],
                "reg_lambda": [0.5, 1.0, 2.0, 5.0],
                "reg_alpha": [0.0, 0.1, 0.5, 1.0],
            }
        if self.name == "lightgbm":
            return {
                "n_estimators": [200, 300, 400, 500],
                "num_leaves": [31, 63, 127, 255],
                "max_depth": [-1, 5, 6, 7, 8],
                "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                "min_child_samples": [10, 20, 30, 50],
                "reg_lambda": [0.5, 1.0, 2.0, 5.0],
                "reg_alpha": [0.0, 0.1, 0.5, 1.0],
            }
        return {}


@dataclass
class AdvancedReport:
    model_name: str
    split: str
    threshold: float
    metrics: dict
    feature_importance: list[dict] | None = None
    best_params: dict | None = None

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "split": self.split,
            "threshold": self.threshold,
            "metrics": self.metrics,
            "feature_importance": self.feature_importance,
            "best_params": self.best_params,
        }


def evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
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
    if metric == "f1":
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
        idx = int(np.nanargmax(f1_scores))
        return float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return 0.5


def train_advanced(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_columns: list[str],
    label_col: str = "failure_in_next_10min",
    model_config: AdvancedModelConfig | None = None,
) -> tuple[XGBClassifier | LGBMClassifier, AdvancedReport]:
    model_config = model_config or AdvancedModelConfig(name="xgboost")
    base_model = model_config.create()

    X_train = train[feature_columns].to_numpy(dtype=float)
    y_train = train[label_col].to_numpy(dtype=int)
    X_val = val[feature_columns].to_numpy(dtype=float)
    y_val = val[label_col].to_numpy(dtype=int)

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    if hasattr(base_model, "set_params"):
        base_model.set_params(scale_pos_weight=float(scale_pos_weight))
    else:
        base_model.scale_pos_weight = float(scale_pos_weight)

    if model_config.tune:
        param_dist = model_config.param_dist()
        cv = StratifiedKFold(n_splits=model_config.tune_cv, shuffle=True, random_state=42)
        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_dist,
            n_iter=model_config.tune_trials,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
        print(f"  Best params: {best_params}")
        print(f"  Best CV AUC: {search.best_score_:.4f}")
    else:
        model = base_model
        model.fit(X_train, y_train)
        best_params = None

    val_scores = model.predict_proba(X_val)[:, 1]
    threshold = find_optimal_threshold(y_val, val_scores)
    val_pred = (val_scores >= threshold).astype(int)

    metrics = evaluate_binary(y_val, val_pred, val_scores)

    fi = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        fi = [
            {"feature": feature_columns[i], "importance": float(importances[i])}
            for i in np.argsort(importances)[::-1]
        ]

    report = AdvancedReport(
        model_name=model_config.name,
        split="val",
        threshold=threshold,
        metrics=metrics,
        feature_importance=fi,
        best_params=best_params,
    )
    return model, report


def evaluate_split(
    model: XGBClassifier | LGBMClassifier,
    df: pd.DataFrame,
    feature_columns: list[str],
    label_col: str,
    threshold: float,
) -> dict:
    X = df[feature_columns].to_numpy(dtype=float)
    y = df[label_col].to_numpy(dtype=int)
    scores = model.predict_proba(X)[:, 1]
    preds = (scores >= threshold).astype(int)
    return evaluate_binary(y, preds, scores)


def save_model(model: XGBClassifier | LGBMClassifier, path: str | Path, threshold: float, feature_columns: list[str], meta: dict | None = None) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path / "model.joblib")
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


def load_model(path: str | Path) -> tuple[XGBClassifier | LGBMClassifier, float, list[str], dict]:
    path = Path(path)
    model = joblib.load(path / "model.joblib")
    with (path / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, float(meta["threshold"]), meta["feature_columns"], meta.get("meta", {})


def build_model_configs(settings: dict) -> dict[str, AdvancedModelConfig]:
    models = settings.get("models", {}).get("advanced", {}) or {}
    return {
        "xgboost": AdvancedModelConfig(
            name="xgboost",
            params=models.get("xgboost", {}),
            tune=models.get("xgboost", {}).get("tune", False),
            tune_trials=models.get("xgboost", {}).get("tune_trials", 30),
            tune_cv=models.get("xgboost", {}).get("tune_cv", 3),
        ),
        "lightgbm": AdvancedModelConfig(
            name="lightgbm",
            params=models.get("lightgbm", {}),
            tune=models.get("lightgbm", {}).get("tune", False),
            tune_trials=models.get("lightgbm", {}).get("tune_trials", 30),
            tune_cv=models.get("lightgbm", {}).get("tune_cv", 3),
        ),
    }