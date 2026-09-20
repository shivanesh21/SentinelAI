from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .simulator.schema import GROUND_TRUTH_HEADER, LOG_HEADER, TELEMETRY_HEADER


class _PartitionWriter:
    def __init__(self, base_dir: Path, header: list[str], fmt: str, flush_every: int, partition_field: str = "timestamp") -> None:
        self.base_dir = base_dir
        self.header = header
        self.fmt = fmt
        self.flush_every = flush_every
        self._partition_field = partition_field
        self._csv_handles: dict[str, Any] = {}
        self._parquet_rows: dict[str, list[dict[str, Any]]] = {}
        self._parquet_seq: dict[str, int] = {}

    def _partition_key(self, ts_iso: str) -> str:
        return "dt=" + ts_iso[:13].replace("T", "-") + ":00:00"

    def _dir_for(self, key: str) -> Path:
        return self.base_dir / key.replace(":", "")

    def write(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        batches: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            batches.setdefault(self._partition_key(str(record[self._partition_field])), []).append(record)
        for key, rows in batches.items():
            dir_path = self._dir_for(key)
            dir_path.mkdir(parents=True, exist_ok=True)
            if self.fmt == "parquet":
                current = self._parquet_rows.setdefault(key, [])
                current.extend(rows)
                if len(current) >= self.flush_every:
                    self._flush_parquet(key, dir_path)
            else:
                handle = self._csv_handles.get(key)
                if handle is None:
                    handle = (dir_path / "data.csv").open("a", newline="", encoding="utf-8")
                    import csv

                    writer = csv.writer(handle, lineterminator="\n")
                    if handle.tell() == 0:
                        writer.writerow(self.header)
                    self._csv_handles[key] = (handle, writer)
                else:
                    handle, writer = handle
                for row in rows:
                    writer.writerow([row.get(col) for col in self.header])

    def _flush_parquet(self, key: str, dir_path: Path) -> None:
        rows = self._parquet_rows[key]
        if not rows:
            return
        self._parquet_seq[key] = self._parquet_seq.get(key, 0) + 1
        frame = pd.DataFrame(rows, columns=self.header)
        frame.to_parquet(dir_path / f"part-{self._parquet_seq[key]:05d}.parquet", index=False)
        self._parquet_rows[key] = []

    def flush_all(self) -> None:
        for key, dir_path in [(k, self._dir_for(k)) for k in list(self._parquet_rows.keys())]:
            self._flush_parquet(key, dir_path)
        for key, (handle, _) in self._csv_handles.items():
            handle.close()
        self._csv_handles = {}


class PartitionedStore:
    def __init__(self, base_dir: Path, fmt: str = "csv", flush_every: int = 64) -> None:
        self.base_dir = Path(base_dir)
        self._metrics = _PartitionWriter(self.base_dir / "raw" / "metrics", TELEMETRY_HEADER, fmt, flush_every)
        self._logs = _PartitionWriter(self.base_dir / "raw" / "logs", LOG_HEADER, fmt, flush_every)
        self._truth = _PartitionWriter(
            self.base_dir / "raw" / "truth", GROUND_TRUTH_HEADER, fmt, flush_every, partition_field="start_ts"
        )

    def append_metrics(self, rows: list[Any]) -> None:
        self._metrics.write([getattr(r, "to_dict")() if not isinstance(r, dict) else r for r in rows])

    def append_logs(self, rows: list[Any]) -> None:
        self._logs.write([getattr(r, "to_dict")() if not isinstance(r, dict) else r for r in rows])

    def append_incidents(self, rows: list[Any]) -> None:
        self._truth.write([getattr(r, "to_dict")() if not isinstance(r, dict) else r for r in rows])

    def close(self) -> None:
        self._metrics.flush_all()
        self._logs.flush_all()
        self._truth.flush_all()