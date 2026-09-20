from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class SplitConfig:
    train_frac: float = 0.6
    val_frac: float = 0.2
    test_frac: float = 0.2
    purge_min: float = 10.0
    interval_sec: int = 30
    time_col: str = "timestamp"

    def purge_ticks(self) -> int:
        return max(0, int(round(self.purge_min * 60.0 / self.interval_sec)))

    def fractions(self) -> tuple[float, float, float]:
        total = self.train_frac + self.val_frac + self.test_frac
        if total <= 0:
            raise ValueError("split fractions must sum to a positive value")
        return self.train_frac / total, self.val_frac / total, self.test_frac / total


@dataclass
class SplitReport:
    purge_ticks: int
    boundaries: dict[str, str] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    services: dict[str, int] = field(default_factory=dict)
    positive_rates: dict[str, dict[str, float]] = field(default_factory=dict)
    overlap_rows: int = 0
    chronological: bool = True

    def to_dict(self) -> dict:
        return {
            "strategy": "time_based",
            "purge_ticks": self.purge_ticks,
            "boundaries": self.boundaries,
            "counts": self.counts,
            "services": self.services,
            "positive_rates": self.positive_rates,
            "overlap_rows": self.overlap_rows,
            "chronological": self.chronological,
        }


def split_config_from_settings(settings: dict, interval_sec: int | None = None) -> SplitConfig:
    splits = settings.get("splits", {}) or {}
    telemetry = settings.get("telemetry", {}) or {}
    return SplitConfig(
        train_frac=float(splits.get("train", 0.6)),
        val_frac=float(splits.get("val", 0.2)),
        test_frac=float(splits.get("test", 0.2)),
        purge_min=float(splits.get("purge_min", telemetry.get("prediction_window_min", 10))),
        interval_sec=int(interval_sec or telemetry.get("collection_interval_sec", 30)),
    )


def time_based_split(
    df: pd.DataFrame,
    config: SplitConfig | None = None,
    label_columns: list[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], SplitReport]:
    """Chronological train/val/test split with a purge gap at each boundary.

    Rows are ordered by timestamp; cut points are computed on the unique timestamp
    grid so every service is split at the same instants. The last ``purge_ticks``
    rows before each boundary are dropped to prevent forward-looking labels from
    leaking across splits.
    """
    cfg = config or SplitConfig()
    if df.empty:
        raise ValueError("cannot split an empty frame")
    if cfg.time_col not in df.columns:
        raise ValueError(f"missing time column: {cfg.time_col}")

    work = df.copy()
    work[cfg.time_col] = pd.to_datetime(work[cfg.time_col], utc=True, format="mixed")
    work = work.sort_values(cfg.time_col, kind="mergesort").reset_index(drop=True)

    timestamps = work[cfg.time_col].drop_duplicates().sort_values().reset_index(drop=True)
    n = len(timestamps)
    if n < 3:
        raise ValueError(f"need at least 3 distinct timestamps to split, got {n}")

    train_frac, val_frac, _ = cfg.fractions()
    cut1 = min(max(1, int(n * train_frac)), n - 2)
    cut2 = min(max(cut1 + 1, int(n * (train_frac + val_frac))), n - 1)

    t1 = timestamps.iloc[cut1 - 1]
    t2 = timestamps.iloc[cut2 - 1]
    gap = pd.Timedelta(seconds=cfg.purge_ticks() * cfg.interval_sec)
    train_end = t1 - gap
    val_end = t2 - gap

    splits = {
        "train": work[work[cfg.time_col] <= train_end],
        "val": work[(work[cfg.time_col] > t1) & (work[cfg.time_col] <= val_end)],
        "test": work[work[cfg.time_col] > t2],
    }

    report = SplitReport(
        purge_ticks=cfg.purge_ticks(),
        boundaries={
            "train_end": _iso(train_end),
            "val_start": _iso(t1),
            "val_end": _iso(val_end),
            "test_start": _iso(t2),
        },
        counts={name: int(len(part)) for name, part in splits.items()},
        services={name: int(part["service"].nunique()) if "service" in part else 0 for name, part in splits.items()},
    )

    labels = label_columns or []
    for name, part in splits.items():
        row = {}
        for label in labels:
            if label in part.columns and len(part):
                row[label] = round(float(part[label].mean()), 6)
        report.positive_rates[name] = row

    report.chronological = _is_chronological(splits, cfg.time_col)
    report.overlap_rows = _overlap_rows(splits, cfg.time_col)
    return splits, report


def _iso(value: pd.Timestamp) -> str:
    return value.isoformat()


def _max_ts(df: pd.DataFrame, time_col: str) -> pd.Timestamp | None:
    return None if df.empty else df[time_col].max()


def _min_ts(df: pd.DataFrame, time_col: str) -> pd.Timestamp | None:
    return None if df.empty else df[time_col].min()


def _is_chronological(splits: dict[str, pd.DataFrame], time_col: str) -> bool:
    order = ["train", "val", "test"]
    previous_max = None
    for name in order:
        current_min = _min_ts(splits[name], time_col)
        if current_min is None:
            continue
        if previous_max is not None and current_min <= previous_max:
            return False
        previous_max = _max_ts(splits[name], time_col)
    return True


def _overlap_rows(splits: dict[str, pd.DataFrame], time_col: str) -> int:
    ranges = [(name, _min_ts(part, time_col), _max_ts(part, time_col)) for name, part in splits.items() if not part.empty]
    overlaps = 0
    for i in range(len(ranges)):
        for j in range(i + 1, len(ranges)):
            _, a_lo, a_hi = ranges[i]
            _, b_lo, b_hi = ranges[j]
            if a_lo <= b_hi and b_lo <= a_hi:
                overlaps += 1
    return overlaps
