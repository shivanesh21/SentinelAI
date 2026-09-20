from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..telemetry.reader import load_raw

ROLLING_METRICS: dict[str, str] = {
    "cpu_util_pct": "CPU",
    "memory_util_pct": "Memory",
    "latency_ms": "Latency",
    "disk_util_pct": "Disk",
    "db_connection_usage_pct": "DB_usage",
    "network_in_mbps": "Network_in",
}

GROWTH_METRICS: dict[str, str] = {
    "cpu_util_pct": "CPU",
    "memory_util_pct": "Memory",
    "disk_util_pct": "Disk",
    "db_connection_usage_pct": "DB_usage",
}

DERIVED_FEATURES: list[str] = [
    "Error_rate",
    "Request_rate",
    "CPU_memory_ratio",
    "latency_per_request",
]


@dataclass
class FeatureConfig:
    interval_sec: int = 30
    windows_min: list[int] = field(default_factory=lambda: [5, 10, 15])
    growth_window_min: int = 10
    error_window_min: int = 5
    request_window_min: int = 5
    min_periods: int = 1
    eps: float = 1e-9
    request_floor_rps: float = 1.0
    passthrough: list[str] = field(default_factory=lambda: ["healthy"])

    def window_ticks(self, minutes: float) -> int:
        return max(1, int(round(minutes * 60.0 / self.interval_sec)))

    @property
    def growth_ticks(self) -> int:
        return self.window_ticks(self.growth_window_min)

    @property
    def error_ticks(self) -> int:
        return self.window_ticks(self.error_window_min)

    @property
    def request_ticks(self) -> int:
        return self.window_ticks(self.request_window_min)


class FeatureEngineer:
    """Causal, per-service rolling and cross-metric feature builder.

    Every feature at time ``t`` uses only the current and past ticks of the same
    service, so the output is safe to use for forecasting without look-ahead
    leakage.
    """

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
        self.rolling_specs: list[tuple[str, str, int]] = [
            (f"{prefix}_{window}min_avg", metric, window)
            for metric, prefix in ROLLING_METRICS.items()
            for window in self.config.windows_min
        ]
        self.growth_specs: list[tuple[str, str, int]] = [
            (f"{prefix}_growth_rate", metric, self.config.growth_window_min)
            for metric, prefix in GROWTH_METRICS.items()
        ]

    @property
    def feature_names(self) -> list[str]:
        return (
            [name for name, _, _ in self.rolling_specs]
            + [name for name, _, _ in self.growth_specs]
            + DERIVED_FEATURES
        )

    def transform(self, metrics: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(metrics)
        df = metrics.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
        df = df.sort_values(["service", "timestamp"], kind="mergesort").reset_index(drop=True)

        out = df[["timestamp", "service"]].copy()
        for name, metric, minutes in self.rolling_specs:
            out[name] = self._rolling_mean(df, metric, self.config.window_ticks(minutes))
        for name, metric, minutes in self.growth_specs:
            out[name] = self._growth_rate(df, metric, self.config.window_ticks(minutes))

        out["Error_rate"] = self._rolling_mean(
            df.assign(_error_rate=df["http_4xx_rate"] + df["http_5xx_rate"]),
            "_error_rate",
            self.config.error_ticks,
        )
        out["Request_rate"] = self._rolling_mean(df, "request_rate_rps", self.config.request_ticks)
        out["CPU_memory_ratio"] = df["cpu_util_pct"] / df["memory_util_pct"].clip(lower=self.config.eps)
        out["latency_per_request"] = df["latency_ms"] / df["request_rate_rps"].clip(
            lower=self.config.request_floor_rps
        )

        for column in self.config.passthrough:
            if column in df.columns:
                out[column] = df[column]

        numeric = [c for c in self.feature_names]
        out[numeric] = out[numeric].astype(float)
        return out

    def _validate_input(self, metrics: pd.DataFrame) -> None:
        required = {"timestamp", "service", "http_4xx_rate", "http_5xx_rate", "request_rate_rps", "cpu_util_pct", "memory_util_pct", "latency_ms"}
        missing = required - set(metrics.columns)
        if missing:
            raise ValueError(f"metrics missing required columns: {sorted(missing)}")
        for metric in set(ROLLING_METRICS) | set(GROWTH_METRICS):
            if metric not in metrics.columns:
                raise ValueError(f"metrics missing required column: {metric}")

    def _rolling_mean(self, df: pd.DataFrame, column: str, ticks: int) -> pd.Series:
        min_periods = min(self.config.min_periods, ticks)
        return (
            df.groupby("service", sort=False)[column]
            .transform(lambda series: series.rolling(ticks, min_periods=min_periods).mean())
        )

    def _growth_rate(self, df: pd.DataFrame, column: str, ticks: int) -> pd.Series:
        current = df[column].astype(float)
        previous = df.groupby("service", sort=False)[column].shift(ticks).astype(float)
        denominator = previous.abs().clip(lower=self.config.eps)
        growth = (current - previous) / denominator
        return growth.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_features(metrics: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    return FeatureEngineer(config).transform(metrics)


def feature_config_from_settings(settings: dict, interval_sec: int | None = None) -> FeatureConfig:
    features = settings.get("features", {}) or {}
    telemetry = settings.get("telemetry", {}) or {}
    return FeatureConfig(
        interval_sec=int(interval_sec or telemetry.get("collection_interval_sec", 30)),
        windows_min=list(features.get("windows_min", telemetry.get("feature_windows_min", [5, 10, 15]))),
        growth_window_min=int(features.get("growth_window_min", 10)),
        error_window_min=int(features.get("error_window_min", 5)),
        request_window_min=int(features.get("request_window_min", 5)),
        min_periods=int(features.get("min_periods", 1)),
        request_floor_rps=float(features.get("request_floor_rps", 1.0)),
    )


def build_features_from_raw(
    root: str | Path,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    metrics = load_raw(root)["metrics"]
    if metrics.empty:
        raise ValueError(f"no metrics found under {Path(root) / 'raw' / 'metrics'}")
    return build_features(metrics, config)
