from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.store import FeatureStore
from src.models.baseline import evaluate_binary
from src.risk import RiskEngine, level_from_score
from src.telemetry.config import load_settings


def main() -> None:
    settings = load_settings(ROOT / "config" / "settings.yaml")
    store = FeatureStore(ROOT / "data" / "features")
    meta = store.load_table_meta()
    feature_columns = meta["feature_columns"]
    label_col = "failure_in_next_10min"

    engine = RiskEngine.load(ROOT / "models", settings)
    engine.feature_columns = feature_columns
    train = store.load_split("train", ROOT / "data" / "splits")
    engine.calibrate(train)
    engine.save_calibration(ROOT / "models" / "risk" / "calibration.json")

    print(f"Components: {engine.available_components()}")
    print(f"Effective weights: {engine._component_weights()}")

    report = {
        "components": {name: engine.weights.get(name) for name in engine.available_components()},
        "effective_weights": engine._component_weights(),
        "severity_thresholds": engine.severity_thresholds,
        "signal_map": engine.signal_map,
        "splits": {},
    }

    for split in ("val", "test"):
        df = store.load_split(split, ROOT / "data" / "splits")
        scored = engine.score(df)
        y = df[label_col].to_numpy()
        flagged = scored["risk"].to_numpy()
        risk_score = scored["risk_score"].to_numpy()
        detection = evaluate_binary(y, flagged, risk_score)

        levels = scored["risk_level"].value_counts()
        level_order = ["NORMAL", "WARNING", "ANOMALOUS", "CRITICAL"]
        level_dist = {name: int(levels.get(name, 0)) for name in level_order}

        tp_windows = flagged & (y == 1)
        lead = df["time_to_failure_min"].to_numpy() if "time_to_failure_min" in df.columns else None
        mean_lead = None
        if lead is not None and tp_windows.any():
            mean_lead = round(float(np.nanmean(lead[tp_windows])), 2)

        report["splits"][split] = {
            "rows": int(len(df)),
            "positive_rate": round(float(y.mean()), 4),
            "level_distribution": level_dist,
            "risk_detection": detection,
            "mean_lead_time_min": mean_lead,
        }
        print(f"--- {split} ---")
        print(f"  levels: {level_dist}")
        print(
            f"  risk: f1={detection['f1']:.4f} precision={detection['precision']:.4f}"
            f" recall={detection['recall']:.4f} roc_auc={detection['roc_auc']:.4f}"
            f" pr_auc={detection['pr_auc']:.4f}"
        )
        if mean_lead is not None:
            print(f"  mean lead time: {mean_lead} min")

    out_path = ROOT / "reports" / "evaluations" / "risk_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()