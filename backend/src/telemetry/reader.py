from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .simulator.schema import GROUND_TRUTH_HEADER, LOG_HEADER, TELEMETRY_HEADER


def _discover_files(location: str | Path, prefix: str) -> list[Path]:
    path = Path(location)
    if path.is_file():
        return [path] if path.name.startswith(prefix) else []
    if path.is_dir():
        return sorted(
            f
            for f in path.rglob("*")
            if f.is_file()
            and (f.name.startswith(prefix) or any(part.startswith("dt=") for part in f.parts))
        )
    return []


def _read_files(files: list[Path], fmt: str | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for file in files:
        ext = fmt or file.suffix.lower()
        if ext == ".csv":
            frames.append(pd.read_csv(file))
        elif ext == ".jsonl":
            frames.append(pd.read_json(file, lines=True))
        elif ext == ".parquet":
            frames.append(pd.read_parquet(file))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def read_metrics(location: str | Path) -> pd.DataFrame:
    df = _read_files(_discover_files(location, "metrics"))
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["timestamp", "service"], keep="last")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
    return df.reset_index(drop=True)


def read_logs(location: str | Path) -> pd.DataFrame:
    df = _read_files(_discover_files(location, "logs"))
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
    return df.reset_index(drop=True)


def read_truth(location: str | Path) -> pd.DataFrame:
    df = _read_files(_discover_files(location, "ground_truth"))
    if df.empty:
        return df
    df["start_ts"] = pd.to_datetime(df["start_ts"], utc=True, format="mixed")
    if "end_ts" in df.columns:
        df["end_ts"] = pd.to_datetime(df["end_ts"], utc=True, format="mixed")
    return df.reset_index(drop=True)


def load_raw(root: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(root)
    metrics = read_metrics(root / "raw" / "metrics")
    logs = read_logs(root / "raw" / "logs")
    truth = read_truth(root / "raw" / "truth")
    if truth.empty:
        truth = read_truth(root / "raw" / "metrics")
    return {"metrics": metrics, "logs": logs, "truth": truth}


def metrics_expected_columns() -> list[str]:
    return TELEMETRY_HEADER


def logs_expected_columns() -> list[str]:
    return LOG_HEADER


def truth_expected_columns() -> list[str]:
    return GROUND_TRUTH_HEADER