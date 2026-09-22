from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class LabelConfig:
    horizons_min: list[int] = field(default_factory=lambda: [10])
    interval_sec: int = 30
    include_current_failure: bool = True
    include_time_to_failure: bool = True

    def window_ticks(self, minutes: float) -> int:
        return max(1, int(round(minutes * 60.0 / self.interval_sec)))

    def label_columns(self) -> list[str]:
        columns = [f"failure_in_next_{h}min" for h in self.horizons_min]
        if self.include_current_failure:
            columns.append("in_failure")
        return columns


def label_columns(config: LabelConfig | None = None) -> list[str]:
    return (config or LabelConfig()).label_columns()


def label_config_from_settings(settings: dict, interval_sec: int | None = None) -> LabelConfig:
    labels = settings.get("labels", {}) or {}
    telemetry = settings.get("telemetry", {}) or {}
    default_horizon = int(telemetry.get("prediction_window_min", 10))
    return LabelConfig(
        horizons_min=list(labels.get("horizons_min", [default_horizon])),
        interval_sec=int(interval_sec or telemetry.get("collection_interval_sec", 30)),
        include_current_failure=bool(labels.get("include_current_failure", True)),
        include_time_to_failure=bool(labels.get("include_time_to_failure", True)),
    )


def build_labels(
    timestamps: pd.DataFrame,
    truth: pd.DataFrame,
    config: LabelConfig | None = None,
) -> pd.DataFrame:
    """Attach failure labels to a per-service timestamp grid.

    ``timestamps`` must contain ``timestamp`` and ``service`` columns. Returns a frame
    with the same rows plus label columns. Labels are causal *targets* (they describe the
    future), so callers must apply a purge/embargo before splitting.
    """
    cfg = config or LabelConfig()
    required = {"timestamp", "service"}
    if not required.issubset(timestamps.columns):
        raise ValueError(f"timestamps missing columns: {sorted(required - set(timestamps.columns))}")

    frame = timestamps[["timestamp", "service"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="mixed")
    frame = frame.sort_values(["service", "timestamp"], kind="mergesort").reset_index(drop=True)

    start_by_service, end_by_service = _window_arrays(truth)

    labels = pd.DataFrame(index=frame.index)
    in_failure_all = np.zeros(len(frame), dtype=bool)
    ttf_all = np.full(len(frame), np.nan, dtype=float)

    for service, idx in frame.groupby("service", sort=False).groups.items():
        idx = np.asarray(idx)
        tick_ns = frame.loc[idx, "timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64")
        starts = start_by_service.get(service, np.array([], dtype="int64"))
        ends = end_by_service.get(service, np.array([], dtype="int64"))

        failing = _in_window(tick_ns, starts, ends)
        in_failure_all[idx] = failing

        for horizon in cfg.horizons_min:
            ticks = cfg.window_ticks(horizon)
            labels.loc[idx, f"failure_in_next_{horizon}min"] = _forward_max(failing, ticks, cfg.include_current_failure)

        if cfg.include_time_to_failure:
            ttf_all[idx] = _time_to_failure(tick_ns, starts, failing)

    if cfg.include_current_failure:
        labels["in_failure"] = in_failure_all.astype(int)
    for horizon in cfg.horizons_min:
        labels[f"failure_in_next_{horizon}min"] = labels[f"failure_in_next_{horizon}min"].astype(int)
    if cfg.include_time_to_failure:
        labels["time_to_failure_min"] = ttf_all

    return pd.concat([frame, labels], axis=1)


def build_feature_labels(
    features: pd.DataFrame,
    truth: pd.DataFrame,
    config: LabelConfig | None = None,
) -> pd.DataFrame:
    labels = build_labels(features[["timestamp", "service"]], truth, config)
    label_cols = [c for c in labels.columns if c not in ("timestamp", "service")]
    return features.merge(labels[["timestamp", "service", *label_cols]], on=["timestamp", "service"], how="left")


def _window_arrays(truth: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    starts: dict[str, np.ndarray] = {}
    ends: dict[str, np.ndarray] = {}
    if truth.empty:
        return starts, ends
    work = truth.copy()
    work["start_ts"] = pd.to_datetime(work["start_ts"], utc=True, format="mixed")
    work["end_ts"] = pd.to_datetime(work["end_ts"].fillna(work["start_ts"]), utc=True, format="mixed")
    for service, group in work.groupby("service"):
        starts[service] = np.sort(group["start_ts"].to_numpy(dtype="datetime64[ns]").astype("int64"))
        ends[service] = np.sort(group["end_ts"].to_numpy(dtype="datetime64[ns]").astype("int64"))
    return starts, ends


def _in_window(tick_ns: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    failing = np.zeros(len(tick_ns), dtype=bool)
    if len(starts) == 0:
        return failing
    for start, end in zip(starts, ends):
        failing |= (tick_ns >= start) & (tick_ns < end)
    return failing


def _forward_max(failing: np.ndarray, ticks: int, include_current: bool) -> np.ndarray:
    series = pd.Series(failing.astype(int))
    forward_inclusive = series[::-1].rolling(ticks, min_periods=1).max()[::-1]
    if include_current:
        return forward_inclusive.to_numpy()
    return forward_inclusive.shift(-1).fillna(0).to_numpy()


def _time_to_failure(tick_ns: np.ndarray, starts: np.ndarray, failing: np.ndarray) -> np.ndarray:
    ttf = np.full(len(tick_ns), np.nan, dtype=float)
    if len(starts):
        idx = np.searchsorted(starts, tick_ns, side="right")
        valid = idx < len(starts)
        next_start = starts[np.clip(idx, 0, len(starts) - 1)]
        minutes = (next_start - tick_ns) / 60_000_000_000.0
        ttf[valid] = minutes[valid]
    ttf[failing] = 0.0
    return ttf
