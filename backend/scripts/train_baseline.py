from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.features.store import FeatureStore
from src.mlops import (
    Experiment,
    ExperimentStore,
    dataset_version,
    registry_from_settings,
    write_stamp,
)
from src.models.baseline import BaselineModelConfig, build_model_configs, evaluate_split, save_model, train_baseline
from src.telemetry.config import load_settings


def main():
    settings = load_settings(ROOT / "config" / "settings.yaml")
    interval_sec = int(settings["telemetry"]["collection_interval_sec"])
    feature_columns = settings["features"].get("feature_columns")
    label_col = "failure_in_next_10min"

    store = FeatureStore(ROOT / "data" / "features")
    train = store.load_split("train", ROOT / "data" / "splits")
    val = store.load_split("val", ROOT / "data" / "splits")
    test = store.load_split("test", ROOT / "data" / "splits")

    if feature_columns is None:
        meta = store.load_table_meta()
        feature_columns = meta["feature_columns"]

    print(f"Feature columns ({len(feature_columns)}): {feature_columns}")
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    print(f"Train positive rate: {train[label_col].mean():.4f}")

    split_report = json.loads((ROOT / "data" / "splits" / "split_report.json").read_text(encoding="utf-8"))
    feature_meta = store.load_table_meta()
    version = dataset_version(feature_meta, split_report)
    registry = ExperimentStore(registry_from_settings(settings, ROOT))
    dataset_summary = {
        "rows": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
        "feature_columns": int(len(feature_columns)),
        "label": label_col,
    }

    model_configs = build_model_configs(settings)

    results = {}
    for name, config in model_configs.items():
        print(f"\n=== Training {name} ===")
        pipe, report = train_baseline(train, val, feature_columns, label_col, config)
        print(f"Val threshold: {report.threshold:.4f}")
        print(f"Val metrics: {report.metrics}")
        if report.feature_importance:
            top5 = report.feature_importance[:5]
            print(f"Top 5 features: {[(f['feature'], round(f['importance'], 4)) for f in top5]}")

        test_metrics = evaluate_split(pipe, test, feature_columns, label_col, report.threshold)
        print(f"Test metrics: {test_metrics}")

        model_dir = ROOT / "models" / "baseline" / name
        save_model(pipe, model_dir, report.threshold, feature_columns, meta={"model_name": name})
        print(f"Saved to {model_dir}")

        experiment = registry.record(
            Experiment(
                family="baseline",
                model=name,
                hyperparameters=pipe.named_steps["clf"].get_params(),
                metrics={
                    "val": report.metrics,
                    "val_threshold": float(report.threshold),
                    "test": test_metrics,
                    "threshold": float(report.threshold),
                },
                dataset_version=version,
                model_version=0,
                model_path=str(model_dir),
                dataset=dataset_summary,
                notes={"top_features": (report.feature_importance[:5] if report.feature_importance else [])},
            )
        )
        write_stamp(model_dir, experiment, registry_from_settings(settings, ROOT))
        print(f"Tracked experiment {experiment.experiment_id} (v{experiment.model_version}) in {registry.registry_path}")

        results[name] = {
            "val": report.to_dict(),
            "test": {"threshold": report.threshold, "metrics": test_metrics},
        }

    report_path = ROOT / "reports" / "evaluations" / "baseline_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()