from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.engineering import (
    build_features_from_raw,
    feature_config_from_settings,
)
from src.telemetry.config import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SentinelAI rolling/cross-metric features")
    parser.add_argument("--root", type=Path, default=ROOT / "data")
    parser.add_argument("--settings", type=Path, default=ROOT / "config" / "settings.yaml")
    parser.add_argument("--interval-sec", type=int, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "features")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    args = parser.parse_args()

    settings = load_settings(args.settings)
    config = feature_config_from_settings(settings, interval_sec=args.interval_sec)
    features = build_features_from_raw(args.root, config)

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"features.{args.format}"
    if args.format == "parquet":
        features.to_parquet(out_path, index=False)
    else:
        features.to_csv(out_path, index=False)

    feature_cols = [c for c in features.columns if c not in ("timestamp", "service", "healthy")]
    per_service = features.groupby("service").size().to_dict()
    nan_total = int(features[feature_cols].isna().sum().sum())
    print(f"done: {len(features)} rows x {len(feature_cols)} features -> {out_path}")
    print(f"per-service rows: {per_service}")
    print(f"NaN in features: {nan_total}")
    print(f"feature columns: {', '.join(feature_cols)}")


if __name__ == "__main__":
    main()
