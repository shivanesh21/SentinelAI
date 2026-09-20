from __future__ import annotations

import os
import pickle
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

from .base import BaseDetector, iter_groups

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.get_logger().setLevel("ERROR")


class _AutoencoderBase(BaseDetector):
    """Shared training/scoring/serialisation for reconstruction-error detectors."""

    kind = "autoencoder"
    persist = "directory"
    normal_labels = ("in_failure", "failure_in_next_10min")

    def __init__(
        self,
        contamination: float = 0.05,
        hidden_dims: tuple[int, ...] = (32, 8),
        epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        seed: int = 42,
        normal_only: bool = True,
        patience: int = 5,
    ):
        super().__init__(contamination=contamination)
        self.hidden_dims = tuple(hidden_dims)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.normal_only = bool(normal_only)
        self.patience = int(patience)
        self.scaler_mean_: np.ndarray | None = None
        self.scaler_std_: np.ndarray | None = None
        self.models_: dict = {}
        self.history_: dict = {}

    def _normal_mask(self, df: pd.DataFrame) -> np.ndarray:
        if not self.normal_only:
            return np.ones(len(df), dtype=bool)
        mask = np.ones(len(df), dtype=bool)
        for column in self.normal_labels:
            if column in df.columns:
                mask &= df[column].to_numpy() == 0
        return mask

    def _build_model(self, input_dim: int) -> keras.Model:
        raise NotImplementedError

    def _fit_scaler(self, matrix: np.ndarray) -> None:
        self.scaler_mean_ = matrix.mean(axis=0)
        self.scaler_std_ = np.maximum(matrix.std(axis=0), self.eps)

    def _scale(self, matrix: np.ndarray) -> np.ndarray:
        return (matrix - self.scaler_mean_) / self.scaler_std_

    def fit(self, df: pd.DataFrame, feature_columns: list[str], group_col: str = "service") -> "_AutoencoderBase":
        keras.utils.set_random_seed(self.seed)
        self.feature_columns = list(feature_columns)
        self.group_col = group_col
        self.models_ = {}
        self.history_ = {}

        normal = df.loc[self._normal_mask(df)]
        if normal.empty:
            normal = df
        self._fit_scaler(normal[self.feature_columns].to_numpy(dtype=float))

        for key, sub in iter_groups(df, group_col):
            mask = self._normal_mask(sub)
            self._fit_group(key if key is not None else "__all__", sub, mask)

        normal_scores = self.score(normal)
        if len(normal_scores):
            self.fit_threshold(normal_scores)
        return self

    def _fit_group(self, key: str, sub: pd.DataFrame, mask: np.ndarray) -> None:
        raise NotImplementedError

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._score_impl(df), dtype=float)

    def _score_impl(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for key, model in self.models_.items():
            model.save(path / f"model_{key.replace('/', '_')}.keras")
        meta = {
            "feature_columns": self.feature_columns,
            "group_col": self.group_col,
            "threshold": self.threshold_,
            "contamination": self.contamination,
            "hidden_dims": self.hidden_dims,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "normal_only": self.normal_only,
            "scaler_mean": self.scaler_mean_,
            "scaler_std": self.scaler_std_,
            "keys": list(self.models_.keys()),
        }
        with (path / "meta.pkl").open("wb") as handle:
            pickle.dump(meta, handle)
        return path

    @classmethod
    def _load_meta(cls, path: Path) -> dict:
        with (path / "meta.pkl").open("rb") as handle:
            return pickle.load(handle)


class DenseAutoencoderDetector(_AutoencoderBase):
    """Fully-connected bottleneck autoencoder; MSE reconstruction error as score."""

    kind = "dense_autoencoder"

    def _build_model(self, input_dim: int) -> keras.Model:
        inputs = keras.Input(shape=(input_dim,))
        x = inputs
        for dim in self.hidden_dims:
            x = layers.Dense(dim, activation="relu")(x)
        for dim in reversed(self.hidden_dims[:-1]):
            x = layers.Dense(dim, activation="relu")(x)
        outputs = layers.Dense(input_dim, activation="linear")(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer=keras.optimizers.Adam(self.learning_rate), loss="mse")
        return model

    def _fit_group(self, key: str, sub: pd.DataFrame, mask: np.ndarray) -> None:
        normal = sub.loc[mask]
        if len(normal) < 10:
            return
        matrix = self._scale(normal[self.feature_columns].to_numpy(dtype=float))
        model = self._build_model(matrix.shape[1])
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=self.patience, restore_best_weights=True, min_delta=1e-5
            )
        ]
        history = model.fit(
            matrix,
            matrix,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            shuffle=True,
            verbose=0,
            callbacks=callbacks,
        )
        self.models_[key] = model
        self.history_[key] = [float(v) for v in history.history.get("val_loss", [])]

    def _score_impl(self, df: pd.DataFrame) -> pd.Series:
        out = pd.Series(0.0, index=df.index)
        fallback = self.models_.get("__all__")
        for key, sub in iter_groups(df, self.group_col):
            model = self.models_.get(key if key is not None else "__all__", fallback)
            if model is None:
                continue
            matrix = self._scale(sub[self.feature_columns].to_numpy(dtype=float))
            reconstruction = model.predict(matrix, batch_size=self.batch_size, verbose=0)
            out.loc[sub.index] = np.mean((matrix - reconstruction) ** 2, axis=1)
        return out

    @classmethod
    def load(cls, path: str | Path) -> "DenseAutoencoderDetector":
        path = Path(path)
        meta = cls._load_meta(path)
        detector = cls(
            contamination=meta["contamination"],
            hidden_dims=tuple(meta["hidden_dims"]),
            epochs=meta["epochs"],
            batch_size=meta["batch_size"],
            learning_rate=meta["learning_rate"],
            seed=meta["seed"],
            normal_only=meta["normal_only"],
        )
        detector.feature_columns = meta["feature_columns"]
        detector.group_col = meta["group_col"]
        detector.threshold_ = meta["threshold"]
        detector.scaler_mean_ = meta["scaler_mean"]
        detector.scaler_std_ = meta["scaler_std"]
        detector.models_ = {
            key: keras.models.load_model(path / f"model_{key.replace('/', '_')}.keras") for key in meta["keys"]
        }
        return detector


class LSTMAutoencoderDetector(_AutoencoderBase):
    """Sequence autoencoder over per-service windows; window MSE attributed to its last tick."""

    kind = "lstm_autoencoder"

    def __init__(self, seq_len: int = 20, lstm_units: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = int(seq_len)
        self.lstm_units = int(lstm_units)

    def _build_model(self, input_dim: int) -> keras.Model:
        inputs = keras.Input(shape=(self.seq_len, input_dim))
        encoded = layers.LSTM(self.lstm_units, activation="tanh")(inputs)
        repeated = layers.RepeatVector(self.seq_len)(encoded)
        decoded = layers.LSTM(self.lstm_units, activation="tanh", return_sequences=True)(repeated)
        outputs = layers.TimeDistributed(layers.Dense(input_dim, activation="linear"))(decoded)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer=keras.optimizers.Adam(self.learning_rate), loss="mse")
        return model

    def _windows(self, matrix: np.ndarray) -> np.ndarray:
        if len(matrix) < self.seq_len:
            return np.empty((0, self.seq_len, matrix.shape[1]), dtype=float)
        indices = np.arange(self.seq_len)[None, :] + np.arange(len(matrix) - self.seq_len + 1)[:, None]
        return matrix[indices]

    def _fit_group(self, key: str, sub: pd.DataFrame, mask: np.ndarray) -> None:
        ordered = sub.sort_values("timestamp") if "timestamp" in sub.columns else sub
        matrix = self._scale(ordered[self.feature_columns].to_numpy(dtype=float))
        labels = ordered[self._label_frame_columns(ordered)].to_numpy() if self.normal_only else None
        windows = self._windows(matrix)
        if len(windows) == 0:
            return
        if labels is not None:
            normal_windows = self._window_mask(labels)
            windows = windows[normal_windows]
        if len(windows) < 10:
            return
        model = self._build_model(matrix.shape[1])
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=self.patience, restore_best_weights=True, min_delta=1e-5
            )
        ]
        history = model.fit(
            windows,
            windows,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            shuffle=True,
            verbose=0,
            callbacks=callbacks,
        )
        self.models_[key] = model
        self.history_[key] = [float(v) for v in history.history.get("val_loss", [])]

    def _label_frame_columns(self, df: pd.DataFrame) -> list[str]:
        return [c for c in self.normal_labels if c in df.columns]

    def _window_mask(self, labels: np.ndarray) -> np.ndarray:
        indices = np.arange(self.seq_len)[None, :] + np.arange(len(labels) - self.seq_len + 1)[:, None]
        normal_per_label = labels[indices].max(axis=1) == 0
        return normal_per_label.all(axis=1)

    def _score_impl(self, df: pd.DataFrame) -> pd.Series:
        out = pd.Series(0.0, index=df.index)
        fallback = self.models_.get("__all__")
        for key, sub in iter_groups(df, self.group_col):
            model = self.models_.get(key if key is not None else "__all__", fallback)
            if model is None:
                continue
            ordered = sub.sort_values("timestamp") if "timestamp" in sub.columns else sub
            matrix = self._scale(ordered[self.feature_columns].to_numpy(dtype=float))
            windows = self._windows(matrix)
            if len(windows) == 0:
                continue
            reconstruction = model.predict(windows, batch_size=self.batch_size, verbose=0)
            errors = np.mean((windows - reconstruction) ** 2, axis=(1, 2))
            scores = np.full(len(matrix), errors[0], dtype=float)
            scores[self.seq_len - 1:] = errors
            out.loc[ordered.index] = scores
        return out

    def save(self, path: str | Path) -> Path:
        path = super().save(path)
        meta = self._load_meta(path)
        meta.update({"seq_len": self.seq_len, "lstm_units": self.lstm_units, "kind": self.kind})
        with (path / "meta.pkl").open("wb") as handle:
            pickle.dump(meta, handle)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "LSTMAutoencoderDetector":
        path = Path(path)
        meta = cls._load_meta(path)
        detector = cls(
            seq_len=meta.get("seq_len", 20),
            lstm_units=meta.get("lstm_units", 32),
            contamination=meta["contamination"],
            hidden_dims=tuple(meta["hidden_dims"]),
            epochs=meta["epochs"],
            batch_size=meta["batch_size"],
            learning_rate=meta["learning_rate"],
            seed=meta["seed"],
            normal_only=meta["normal_only"],
        )
        detector.feature_columns = meta["feature_columns"]
        detector.group_col = meta["group_col"]
        detector.threshold_ = meta["threshold"]
        detector.scaler_mean_ = meta["scaler_mean"]
        detector.scaler_std_ = meta["scaler_std"]
        detector.models_ = {
            key: keras.models.load_model(path / f"model_{key.replace('/', '_')}.keras") for key in meta["keys"]
        }
        return detector
