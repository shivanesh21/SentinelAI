from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from src.anomaly import (
    DenseAutoencoderDetector,
    IQRTukeyDetector,
    IsolationForestDetector,
    LSTMAutoencoderDetector,
    RollingIQRDetector,
    ZScoreDetector,
    assign_levels,
)
from src.features.store import FeatureStore
from src.telemetry.config import load_settings


def detector_zoo(settings: dict, interval_sec: int, include_lstm: bool = False) -> dict:
    anomaly = settings.get("anomaly", {}) or {}
    contamination = float(anomaly.get("contamination", 0.05))
    forest = anomaly.get("isolation_forest", {}) or {}
    ae = anomaly.get("autoencoder", {}) or {}
    ae_kwargs = dict(
        contamination=contamination,
        hidden_dims=tuple(ae.get("hidden_dims", [32, 8])),
        epochs=int(ae.get("epochs", 50)),
        batch_size=int(ae.get("batch_size", 256)),
        learning_rate=float(ae.get("learning_rate", 1e-3)),
        seed=int(ae.get("seed", 42)),
        normal_only=bool(ae.get("normal_only", True)),
    )
    zoo = {
        "zscore": ZScoreDetector(contamination=contamination),
        "iqr": IQRTukeyDetector(contamination=contamination),
        "rolling_iqr": RollingIQRDetector(
            contamination=contamination,
            iqr_window_min=float(anomaly.get("rolling_iqr_window_min", 60)),
            interval_sec=interval_sec,
        ),
        "isolation_forest": IsolationForestDetector(
            contamination=contamination,
            n_estimators=int(forest.get("n_estimators", 200)),
            random_state=int(forest.get("random_state", 42)),
        ),
        "autoencoder": DenseAutoencoderDetector(**ae_kwargs),
    }
    if include_lstm or bool(ae.get("lstm_enabled", False)):
        zoo["lstm_autoencoder"] = LSTMAutoencoderDetector(
            seq_len=int(ae.get("seq_len", 20)),
            lstm_units=int(ae.get("lstm_units", 32)),
            **ae_kwargs,
        )
    return zoo


def classification_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = (scores > threshold).astype(int)
    metrics = {
        "threshold": round(float(threshold), 6),
        "positives": int(y_true.sum()),
        "predicted_positives": int(predicted.sum()),
        "precision": round(float(precision_score(y_true, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predicted, zero_division=0)), 4),
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, scores)), 4)
        metrics["pr_auc"] = round(float(average_precision_score(y_true, scores)), 4)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit anomaly baselines and score validation/test data")
    parser.add_argument("--settings", type=Path, default=ROOT / "config" / "settings.yaml")
    parser.add_argument("--features-dir", type=Path, default=ROOT / "data" / "features")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--scores-out", type=Path, default=ROOT / "data" / "features" / "anomaly_scores.parquet")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "evaluations" / "anomaly_baseline_report.json")
    parser.add_argument("--interval-sec", type=int, default=None)
    parser.add_argument("--include-lstm", action="store_true", help="also train the LSTM autoencoder")
    args = parser.parse_args()

    settings = load_settings(args.settings)
    telemetry = settings.get("telemetry", {})
    interval_sec = int(args.interval_sec or telemetry.get("collection_interval_sec", 30))
    anomaly_cfg = settings.get("anomaly", {}) or {}
    levels = tuple(anomaly_cfg.get("levels", ["NORMAL", "WARNING", "ANOMALOUS", "CRITICAL"]))
    contamination = float(anomaly_cfg.get("contamination", 0.05))

    store = FeatureStore(args.features_dir)
    meta = store.read_metadata()
    feature_columns = meta.get("feature_columns", [])
    if not feature_columns:
        raise SystemExit("feature table metadata missing feature_columns; run scripts/build_dataset.py first")

    splits = store.read_splits(args.splits_dir)
    train, val = splits["train"], splits["val"]
    test = splits.get("test")
    label = "failure_in_next_10min"

    zoo = detector_zoo(settings, interval_sec, include_lstm=args.include_lstm)
    report: dict = {"label": label, "contamination": contamination, "feature_count": len(feature_columns), "models": {}}
    score_frames: list[pd.DataFrame] = []

    print(f"features: {len(feature_columns)}   train={len(train)}  val={len(val)}  test={len(test) if test is not None else 0}")
    for name, detector in zoo.items():
        detector.fit(train, feature_columns)
        train_scores = detector.score(train)
        detector.fit_threshold(train_scores)

        entry = {"threshold": round(float(detector.threshold_), 6), "splits": {}}
        for split_name, split_df in (("val", val), ("test", test)):
            if split_df is None:
                continue
            scores = detector.score(split_df)
            y_true = split_df[label].to_numpy()
            entry["splits"][split_name] = classification_metrics(y_true, scores, detector.threshold_)

            frame = split_df[["timestamp", "service"]].copy()
            frame["model"] = name
            frame["split"] = split_name
            frame["score"] = scores
            frame["anomaly"] = (scores > detector.threshold_).astype(int)
            frame["level"] = assign_levels(scores, train_scores, levels=levels, contamination=contamination)
            frame["failure_in_next_10min"] = y_true
            score_frames.append(frame)

        val_metrics = entry["splits"].get("val", {})
        print(f"{name:<17} val roc_auc={val_metrics.get('roc_auc')} pr_auc={val_metrics.get('pr_auc')} "
              f"prec={val_metrics.get('precision')} rec={val_metrics.get('recall')} f1={val_metrics.get('f1')}")
        report["models"][name] = entry

    scores_df = pd.concat(score_frames, ignore_index=True)
    args.scores_out.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_parquet(args.scores_out, index=False)

    report["scored_rows"] = int(len(scores_df))
    report["scores_output"] = str(args.scores_out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"anomaly scores -> {args.scores_out}  ({len(scores_df)} rows)")
    print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
