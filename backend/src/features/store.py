from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCHEMA_VERSION = "1.0"
DEFAULT_TABLE_NAME = "feature_table"


@dataclass
class FeatureTableMeta:
    schema_version: str = SCHEMA_VERSION
    created_utc: str = ""
    interval_sec: int = 30
    prediction_window_min: int = 10
    feature_columns: list[str] = field(default_factory=list)
    label_columns: list[str] = field(default_factory=list)
    n_rows: int = 0
    n_services: int = 0
    services: list[str] = field(default_factory=list)
    start_ts: str = ""
    end_ts: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class FeatureStore:
    """Simple parquet-backed feature store with JSON sidecar metadata."""

    def __init__(self, base_dir: str | Path = "data/features"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def table_path(self, name: str = DEFAULT_TABLE_NAME) -> Path:
        return self.base_dir / f"{name}.parquet"

    def meta_path(self, name: str = DEFAULT_TABLE_NAME) -> Path:
        return self.base_dir / f"{name}.meta.json"

    def write_table(
        self,
        df: pd.DataFrame,
        name: str = DEFAULT_TABLE_NAME,
        metadata: FeatureTableMeta | None = None,
    ) -> Path:
        path = self.table_path(name)
        df.to_parquet(path, index=False)
        meta = metadata or self._infer_metadata(df)
        self.meta_path(name).write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")
        return path

    def read_table(self, name: str = DEFAULT_TABLE_NAME) -> pd.DataFrame:
        path = self.table_path(name)
        if not path.exists():
            raise FileNotFoundError(f"feature table not found: {path}")
        return pd.read_parquet(path)

    def read_metadata(self, name: str = DEFAULT_TABLE_NAME) -> dict:
        path = self.meta_path(name)
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def write_splits(
        self,
        splits: dict[str, pd.DataFrame],
        out_dir: str | Path = "data/splits",
        report: dict | None = None,
    ) -> dict[str, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        written = {}
        for name, part in splits.items():
            path = out / f"{name}.parquet"
            part.to_parquet(path, index=False)
            written[name] = path
        if report is not None:
            (out / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return written

    def read_splits(self, out_dir: str | Path = "data/splits") -> dict[str, pd.DataFrame]:
        base = Path(out_dir)
        return {name: pd.read_parquet(base / f"{name}.parquet") for name in ("train", "val", "test")}

    def load_split(self, name: str, out_dir: str | Path = "data/splits") -> pd.DataFrame:
        path = Path(out_dir) / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"split not found: {path}")
        return pd.read_parquet(path)

    def load_table_meta(self, name: str = DEFAULT_TABLE_NAME) -> dict:
        path = self.meta_path(name)
        if not path.exists():
            raise FileNotFoundError(f"table metadata not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _infer_metadata(self, df: pd.DataFrame) -> FeatureTableMeta:
        label_cols = sorted(
            c for c in df.columns if c.startswith("failure_in_next_") or c in ("in_failure", "healthy")
        )
        excluded = {"timestamp", "service", "time_to_failure_min", *label_cols}
        feature_cols = [c for c in df.columns if c not in excluded]
        ts = pd.to_datetime(df["timestamp"], utc=True, format="mixed") if "timestamp" in df else pd.Series([], dtype="datetime64[ns, UTC]")
        return FeatureTableMeta(
            created_utc=datetime.now(timezone.utc).isoformat(),
            feature_columns=feature_cols,
            label_columns=label_cols,
            n_rows=int(len(df)),
            n_services=int(df["service"].nunique()) if "service" in df else 0,
            services=sorted(df["service"].unique().tolist()) if "service" in df else [],
            start_ts=ts.min().isoformat() if len(ts) else "",
            end_ts=ts.max().isoformat() if len(ts) else "",
        )


def build_feature_table_metadata(
    df: pd.DataFrame,
    feature_columns: list[str],
    label_columns: list[str],
    interval_sec: int,
    prediction_window_min: int,
) -> FeatureTableMeta:
    ts = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
    return FeatureTableMeta(
        created_utc=datetime.now(timezone.utc).isoformat(),
        interval_sec=interval_sec,
        prediction_window_min=prediction_window_min,
        feature_columns=list(feature_columns),
        label_columns=list(label_columns),
        n_rows=int(len(df)),
        n_services=int(df["service"].nunique()),
        services=sorted(df["service"].unique().tolist()),
        start_ts=ts.min().isoformat(),
        end_ts=ts.max().isoformat(),
    )
