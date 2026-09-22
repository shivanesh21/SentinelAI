from __future__ import annotations

import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.get_logger().setLevel("ERROR")

from .baseline import evaluate_binary, find_optimal_threshold


def build_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    label_col: str = "failure_in_next_10min",
    seq_len: int = 20,
    group_col: str = "service",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Slide non-overlapping-by-one windows inside each service's time series.

    Windows never cross service boundaries and the split's leading ``seq_len - 1`` warm-up
    rows per service are dropped, so training windows contain no future labels.
    Returns ``(X, y, meta)`` where ``meta`` holds timestamp/service of each window's last row.
    """
    arrays: list[np.ndarray] = []
    labels: list[int] = []
    meta_rows: list[dict] = []
    for _, group in df.groupby(group_col, sort=True):
        if len(group) < seq_len:
            continue
        group = group.sort_values("timestamp")
        features = group[feature_columns].to_numpy(dtype=float)
        y = group[label_col].to_numpy()
        matches = group[["timestamp", "service"]].to_dict("records")
        for end in range(seq_len - 1, len(group)):
            arrays.append(features[end - seq_len + 1 : end + 1])
            labels.append(int(y[end]))
            meta_rows.append(matches[end])
    X = np.stack(arrays) if arrays else np.empty((0, seq_len, len(feature_columns)), dtype=float)
    y = np.asarray(labels, dtype=int)
    meta = pd.DataFrame(meta_rows)
    return X, y, meta


def build_sequences_for_splits(
    splits: dict[str, pd.DataFrame],
    feature_columns: list[str],
    label_col: str = "failure_in_next_10min",
    seq_len: int = 20,
    group_col: str = "service",
) -> dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
    return {
        name: build_sequences(df, feature_columns, label_col, seq_len, group_col)
        for name, df in splits.items()
    }


class _FeatureScaler:
    def __init__(self, epsilon: float = 1e-8):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.epsilon = float(epsilon)

    def fit(self, X: np.ndarray) -> "_FeatureScaler":
        flat = X.reshape(-1, X.shape[-1])
        self.mean_ = flat.mean(axis=0)
        self.std_ = np.maximum(flat.std(axis=0), self.epsilon)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_


@dataclass
class SequentialConfig:
    seq_len: int = 20
    lstm_units: int = 32
    dropout: float = 0.2
    learning_rate: float = 1e-3
    epochs: int = 50
    batch_size: int = 128
    patience: int = 5
    seed: int = 42


def sequential_config_from_settings(settings: dict) -> SequentialConfig:
    block = settings.get("models", {}).get("sequential", {}) or {}
    valid = {field.name for field in SequentialConfig.__dataclass_fields__.values() if field.name != "seq_len"}
    kwargs = {k: v for k, v in block.items() if k in valid}
    return SequentialConfig(**kwargs)


def sequential_configs_from_settings(
    settings: dict,
) -> tuple[SequentialConfig, bool, list[SequentialConfig]]:
    block = settings.get("models", {}).get("sequential", {}) or {}
    default = sequential_config_from_settings(settings)
    tune = bool(block.get("tune", False))
    candidates = [SequentialConfig(**item) for item in block.get("tune_space", [])]
    return default, tune, candidates


def tune_sequential(
    splits: dict[str, pd.DataFrame],
    feature_columns: list[str],
    configs: list[SequentialConfig],
    label_col: str = "failure_in_next_10min",
    group_col: str = "service",
) -> tuple[SequentialConfig, list[dict]]:
    """Rebuild windows per candidate seq_len, fit on train, rank by validation F1."""
    results: list[dict] = []
    best_config = configs[0]
    best_f1 = -1.0
    for cfg in configs:
        X_train, y_train, _ = build_sequences(
            splits["train"], feature_columns, label_col, cfg.seq_len, group_col
        )
        X_val, y_val, _ = build_sequences(
            splits["val"], feature_columns, label_col, cfg.seq_len, group_col
        )
        model = SequentialLSTMClassifier(cfg).fit(X_train, y_train, X_val, y_val)
        evaluation = model.evaluate(X_val, y_val)
        records = {
            "config": asdict(cfg),
            "epochs_run": len(model.history_.get("auc", [])),
            "val_f1": round(float(evaluation["metrics"]["f1"]), 4),
            "val_roc_auc": round(float(evaluation["metrics"]["roc_auc"]), 4),
            "val_threshold": float(evaluation["threshold"]),
        }
        results.append(records)
        print(
            f"  {cfg} f1={records['val_f1']} roc_auc={records['val_roc_auc']}"
            f" epochs={records['epochs_run']}"
        )
        if records["val_f1"] > best_f1:
            best_f1 = records["val_f1"]
            best_config = cfg
    return best_config, results


class SequentialLSTMClassifier:
    """LSTM binary classifier over per-service windows of the feature table."""

    def __init__(self, config: SequentialConfig | None = None):
        self.config = config or SequentialConfig()
        self.scaler = _FeatureScaler()
        self.model: keras.Model | None = None
        self.class_weight_: dict[int, float] = {}
        self.history_: dict[str, list[float]] = {}
        self.n_features_: int = 0

    def _build(self, n_features: int) -> keras.Model:
        keras.utils.set_random_seed(self.config.seed)
        inp = layers.Input(shape=(self.config.seq_len, n_features))
        x = layers.LSTM(self.config.lstm_units, dropout=self.config.dropout, return_sequences=False)(inp)
        x = layers.Dense(16, activation="relu")(x)
        x = layers.Dropout(self.config.dropout)(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inp, out)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.AUC(name="auc")],
        )
        return model

    def _set_class_weight(self, y: np.ndarray) -> None:
        n0 = int((y == 0).sum())
        n1 = int((y == 1).sum())
        self.class_weight_ = {0: 1.0, 1: n0 / n1 if n1 and n0 else 1.0}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "SequentialLSTMClassifier":
        self.n_features_ = int(X_train.shape[-1])
        self.scaler.fit(X_train)
        self._set_class_weight(y_train)
        self.model = self._build(self.n_features_)
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=self.config.patience, restore_best_weights=True
            )
        ]
        history = self.model.fit(
            self.scaler.transform(X_train),
            y_train,
            validation_data=(self.scaler.transform(X_val), y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            class_weight=self.class_weight_,
            callbacks=callbacks,
            verbose=0,
        )
        self.history_ = {k: [float(v) for v in history.history[k]] for k in history.history}
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X), verbose=0).ravel()

    def evaluate(self, X: np.ndarray, y: np.ndarray, threshold: float | None = None) -> dict:
        scores = self.predict_proba(X)
        if threshold is None:
            threshold = find_optimal_threshold(y, scores)
        preds = (scores >= threshold).astype(int)
        return {"threshold": float(threshold), "metrics": evaluate_binary(y, preds, scores)}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(path / "model.keras")
        payload = {
            "config": asdict(self.config),
            "n_features": self.n_features_,
            "class_weight": self.class_weight_,
            "history": self.history_,
            "scaler_mean": self.scaler.mean_,
            "scaler_std": self.scaler.std_,
        }
        with (path / "meta.pkl").open("wb") as handle:
            pickle.dump(payload, handle)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SequentialLSTMClassifier":
        path = Path(path)
        with (path / "meta.pkl").open("rb") as handle:
            payload = pickle.load(handle)
        config = SequentialConfig(**payload["config"])
        model = cls(config)
        model.n_features_ = int(payload["n_features"])
        model.class_weight_ = payload["class_weight"]
        model.history_ = payload["history"]
        model.scaler.mean_ = payload["scaler_mean"]
        model.scaler.std_ = payload["scaler_std"]
        model.model = keras.models.load_model(path / "model.keras")
        return model