from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_REGISTRY_PATH = "data/experiments.jsonl"
FAMILIES = ("baseline", "advanced", "sequential", "risk")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def dataset_version(feature_meta: dict | None = None, split_report: dict | None = None) -> str:
    payload = {
        "features": {
            k: (feature_meta or {}).get(k)
            for k in (
                "name",
                "n_rows",
                "n_services",
                "feature_columns",
                "start_ts",
                "end_ts",
                "prediction_window_min",
                "interval_sec",
            )
        },
        "splits": split_report or {},
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return f"ds-{digest}"


def registry_from_settings(settings: dict, base: Path) -> Path:
    rel = settings.get("mlops", {}).get("registry", DEFAULT_REGISTRY_PATH)
    path = Path(rel)
    return path if path.is_absolute() else base / path


@dataclass
class Experiment:
    family: str
    model: str
    hyperparameters: dict
    metrics: dict
    dataset_version: str = ""
    model_version: int = 0
    model_path: str = ""
    experiment_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    trained_at: str = field(default_factory=utcnow)
    status: str = "ok"
    dataset: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "family": self.family,
            "model": self.model,
            "model_version": self.model_version,
            "dataset_version": self.dataset_version,
            "dataset": self.dataset,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "model_path": self.model_path,
            "trained_at": self.trained_at,
            "status": self.status,
            "notes": self.notes,
        }


class ExperimentStore:
    def __init__(self, registry_path: str | Path = DEFAULT_REGISTRY_PATH) -> None:
        self.registry_path = Path(registry_path)

    def next_version(self, family: str, model: str) -> int:
        records = [r for r in self.read() if r["family"] == family and r["model"] == model]
        return (max(r["model_version"] for r in records) + 1) if records else 1

    def record(self, experiment: Experiment) -> Experiment:
        if not experiment.model_version:
            experiment.model_version = self.next_version(experiment.family, experiment.model)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with self.registry_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(experiment.to_dict(), default=str) + "\n")
        return experiment

    def read(
        self,
        limit: int = 200,
        family: str | None = None,
        model: str | None = None,
    ) -> list[dict]:
        if not self.registry_path.exists():
            return []
        records = []
        with self.registry_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if family:
            records = [r for r in records if r["family"] == family]
        if model:
            records = [r for r in records if r["model"] == model]
        if limit <= 0:
            return []
        records = records[-limit:]
        return records

    def get(self, experiment_id: str) -> dict | None:
        for record in self.read():
            if record["experiment_id"] == experiment_id:
                return record
        return None

    def latest(self, family: str, model: str) -> dict | None:
        records = self.read(family=family, model=model)
        return records[-1] if records else None

    def best(self, family: str, model: str, metric: str = "f1", split: str = "test") -> dict | None:
        scored = [
            (record, record["metrics"].get(split, {}).get(metric))
            for record in self.read(family=family, model=model)
            if record["status"] == "ok"
        ]
        scored = [(r, v) for r, v in scored if v is not None]
        return max(scored, key=lambda pair: pair[1])[0] if scored else None

    def models(self) -> list[str]:
        return sorted({f"{r['family']}/{r['model']}" for r in self.read()})

    def compare(self, metric: str = "f1", split: str = "test") -> list[dict]:
        rows = []
        for key in self.models():
            family, model = key.split("/", 1)
            best = self.best(family, model, metric, split)
            if best:
                rows.append(
                    {
                        "family": family,
                        "model": model,
                        "model_version": best["model_version"],
                        "dataset_version": best["dataset_version"],
                        "split": split,
                        "metric": metric,
                        "value": best["metrics"].get(split, {}).get(metric),
                        "metrics": best["metrics"].get(split, {}),
                        "experiment_id": best["experiment_id"],
                        "trained_at": best["trained_at"],
                    }
                )
        return sorted(rows, key=lambda r: (-(r["value"] or 0), r["model"]))


def write_stamp(
    model_dir: str | Path,
    experiment: Experiment,
    registry_path: str | Path | None = None,
) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    stamp = {
        "experiment_id": experiment.experiment_id,
        "family": experiment.family,
        "model": experiment.model,
        "model_version": experiment.model_version,
        "dataset_version": experiment.dataset_version,
        "trained_at": experiment.trained_at,
        "registry": str(registry_path) if registry_path else None,
    }
    path = model_dir / "mlops.json"
    path.write_text(json.dumps(stamp, indent=2, default=str), encoding="utf-8")
    return path