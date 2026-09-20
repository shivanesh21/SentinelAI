from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.engineering import build_features_from_raw, feature_config_from_settings
from src.features.labels import build_labels, label_config_from_settings
from src.features.splits import split_config_from_settings, time_based_split
from src.features.store import FeatureStore, build_feature_table_metadata
from src.telemetry.config import load_settings
from src.telemetry.reader import load_raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the labeled SentinelAI feature table and time-based splits")
    parser.add_argument("--root", type=Path, default=ROOT / "data")
    parser.add_argument("--settings", type=Path, default=ROOT / "config" / "settings.yaml")
    parser.add_argument("--features-dir", type=Path, default=ROOT / "data" / "features")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "evaluations" / "dataset_report.json")
    parser.add_argument("--interval-sec", type=int, default=None)
    args = parser.parse_args()

    settings = load_settings(args.settings)
    telemetry = settings.get("telemetry", {})
    interval_sec = int(args.interval_sec or telemetry.get("collection_interval_sec", 30))

    raw = load_raw(args.root)
    metrics, truth = raw["metrics"], raw["truth"]
    if metrics.empty:
        raise SystemExit(f"no metrics found under {args.root / 'raw' / 'metrics'}")

    features = build_features_from_raw(args.root, feature_config_from_settings(settings, interval_sec))
    labels = build_labels(metrics[["timestamp", "service"]], truth, label_config_from_settings(settings, interval_sec))
    label_cols = [c for c in labels.columns if c not in ("timestamp", "service")]
    table = features.merge(labels, on=["timestamp", "service"], how="left")

    feature_cols = [c for c in features.columns if c not in ("timestamp", "service", "healthy")]
    metadata = build_feature_table_metadata(
        table,
        feature_columns=feature_cols,
        label_columns=[c for c in label_cols if c != "time_to_failure_min"],
        interval_sec=interval_sec,
        prediction_window_min=int(telemetry.get("prediction_window_min", 10)),
    )
    store = FeatureStore(args.features_dir)
    table_path = store.write_table(table, metadata=metadata)

    target_labels = [c for c in label_cols if c.startswith("failure_in_next_")]
    splits, split_report = time_based_split(table, split_config_from_settings(settings, interval_sec), target_labels)
    written = store.write_splits(splits, args.splits_dir, report=split_report.to_dict())

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps({"feature_table": metadata.to_dict(), "splits": split_report.to_dict()}, indent=2),
        encoding="utf-8",
    )

    print(f"feature table: {len(table)} rows x {len(feature_cols)} features + {len(label_cols)} labels -> {table_path}")
    print(f"labels: {label_cols}")
    for label in target_labels:
        rate = float(table[label].mean())
        print(f"  {label}: {int(table[label].sum())} positives ({rate:.4f})")
    print(f"splits: {split_report.counts}  purge={split_report.purge_ticks} ticks  chronological={split_report.chronological}")
    print(f"boundaries: {split_report.boundaries}")
    for name, part in splits.items():
        print(f"  {name}: {len(part)} rows, {part['service'].nunique()} services, pos={split_report.positive_rates[name]}")
    print(f"split files -> {[str(p) for p in written.values()]}")
    print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
