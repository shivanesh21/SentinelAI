from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mlops import DriftMonitoringEngine
from src.telemetry.config import load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SentinelAI Drift Detection & Performance Monitoring")
    parser.add_argument(
        "--reference",
        type=str,
        default="data/splits/train.parquet",
        help="Path to baseline/reference dataset (default: data/splits/train.parquet)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="data/splits/test.parquet",
        help="Path to target/production dataset (default: data/splits/test.parquet)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/baseline/random_forest",
        help="Path to trained model directory containing model.joblib and meta.json",
    )
    parser.add_argument(
        "--simulate-drift",
        action="store_true",
        help="Inject synthetic feature distribution shift to test retraining trigger",
    )
    parser.add_argument(
        "--simulate-degradation",
        action="store_true",
        help="Inject model prediction errors to test performance degradation trigger",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom path to save the JSON drift monitoring report",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix in (".csv", ".txt"):
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")


def apply_simulated_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Inject distribution drift on selected features to simulate real-world service anomalies."""
    df_drift = df.copy()
    np.random.seed(42)

    # Shift latency and error rate distributions
    if "Latency_15min_avg" in df_drift.columns:
        df_drift["Latency_15min_avg"] = df_drift["Latency_15min_avg"] * 3.5 + np.random.exponential(150, len(df_drift))
    if "Latency_5min_avg" in df_drift.columns:
        df_drift["Latency_5min_avg"] = df_drift["Latency_5min_avg"] * 3.0 + np.random.exponential(120, len(df_drift))
    if "Latency_10min_avg" in df_drift.columns:
        df_drift["Latency_10min_avg"] = df_drift["Latency_10min_avg"] * 3.2 + np.random.exponential(135, len(df_drift))
    if "Error_rate" in df_drift.columns:
        df_drift["Error_rate"] = np.clip(df_drift["Error_rate"] * 5.0 + np.random.beta(2, 5, len(df_drift)) * 0.4, 0.0, 1.0)
    if "Memory_growth_rate" in df_drift.columns:
        df_drift["Memory_growth_rate"] = df_drift["Memory_growth_rate"] * 4.0 + 0.15
    if "Memory_15min_avg" in df_drift.columns:
        df_drift["Memory_15min_avg"] = np.clip(df_drift["Memory_15min_avg"] + 35.0, 0.0, 100.0)
    if "DB_usage_15min_avg" in df_drift.columns:
        df_drift["DB_usage_15min_avg"] = np.clip(df_drift["DB_usage_15min_avg"] * 2.2 + 20.0, 0.0, 100.0)
    if "CPU_growth_rate" in df_drift.columns:
        df_drift["CPU_growth_rate"] = df_drift["CPU_growth_rate"] + np.random.normal(0.2, 0.05, len(df_drift))

    return df_drift


def predict_model(model_dir: Path, target_df: pd.DataFrame, feature_columns: list[str]) -> tuple[np.ndarray | None, float | None]:
    model_path = model_dir / "model.joblib"
    meta_path = model_dir / "meta.json"
    if not model_path.exists():
        return None, None

    try:
        model = joblib.load(model_path)
        threshold = 0.5
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            threshold = float(meta.get("threshold", 0.5))

        X = target_df[feature_columns]
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
            preds = (probs >= threshold).astype(int)
        else:
            preds = model.predict(X).astype(int)
        return preds, threshold
    except Exception as exc:
        print(f"Warning: could not evaluate model: {exc}")
        return None, None


def main() -> None:
    args = parse_args()
    settings = load_settings(ROOT / "config" / "settings.yaml")
    engine = DriftMonitoringEngine(settings=settings, base_dir=ROOT)

    ref_path = ROOT / args.reference if not Path(args.reference).is_absolute() else Path(args.reference)
    target_path = ROOT / args.target if not Path(args.target).is_absolute() else Path(args.target)
    model_dir = ROOT / args.model_dir if not Path(args.model_dir).is_absolute() else Path(args.model_dir)

    print("=" * 82)
    print("SENTINELAI DRIFT DETECTION & PERFORMANCE MONITORING (DAY 22)")
    print("=" * 82)
    print(f"Reference dataset : {ref_path}")
    print(f"Target dataset    : {target_path}")

    ref_df = load_dataset(ref_path)
    target_df = load_dataset(target_path)

    if args.simulate_drift:
        print(">> Simulating severe feature drift on target dataset...")
        target_df = apply_simulated_drift(target_df)

    # Determine feature columns
    meta_path = ROOT / "data" / "features" / "feature_table.meta.json"
    feature_columns = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        feature_columns = [c for c in meta.get("feature_columns", []) if c in ref_df.columns and c in target_df.columns]

    # Predict using baseline model if available
    y_true = None
    y_pred = None
    baseline_f1 = None

    if "failure_in_next_10min" in target_df.columns and feature_columns:
        y_true = target_df["failure_in_next_10min"].values
        preds, threshold = predict_model(model_dir, target_df, feature_columns)
        if preds is not None:
            y_pred = preds
            if args.simulate_degradation:
                print(">> Simulating model degradation (corrupting predictions)...")
                # Flip random 40% of positive predictions to false negatives
                mask = (y_true == 1) & (np.random.rand(len(y_true)) < 0.7)
                y_pred[mask] = 0

            # Fetch baseline test F1 from reports if available
            baseline_report_path = ROOT / "reports" / "evaluations" / "baseline_report.json"
            if baseline_report_path.exists():
                b_rep = json.loads(baseline_report_path.read_text(encoding="utf-8"))
                baseline_f1 = b_rep.get("random_forest", {}).get("test", {}).get("metrics", {}).get("f1")
            if baseline_f1 is None:
                baseline_f1 = 0.8500  # nominal baseline F1

    report = engine.run(
        reference_df=ref_df,
        target_df=target_df,
        feature_columns=feature_columns,
        y_true=y_true,
        y_pred=y_pred,
        baseline_f1=baseline_f1,
        metadata={
            "reference_file": str(ref_path),
            "target_file": str(target_path),
            "simulated_drift": args.simulate_drift,
            "simulated_degradation": args.simulate_degradation,
        },
    )

    out_path = engine.save_report(report, path=args.output)

    # Render summary tables
    print("\n" + "-" * 82)
    print("FEATURE DRIFT ANALYSIS (PSI & Two-Sample KS-Test)")
    print("-" * 82)
    fmt = "{:<25s} | {:>7s} | {:>7s} | {:>9s} | {:<12s} | {:<8s}"
    print(fmt.format("Feature", "PSI", "KS-Stat", "KS p-val", "Level", "Drift?"))
    print("-" * 82)

    for feat, res in sorted(report.feature_metrics.items(), key=lambda x: -x[1].psi):
        d_flag = "[DRIFT]" if res.drift_detected else "OK"
        print(fmt.format(
            feat[:25],
            f"{res.psi:.4f}",
            f"{res.ks_statistic:.4f}",
            f"{res.ks_pvalue:.4f}",
            res.drift_level,
            d_flag,
        ))

    summary = report.data_drift
    print("-" * 82)
    print(f"Summary: {summary.drifted_features_count}/{summary.total_features} features drifted "
          f"({summary.drift_share * 100:.1f}% share, threshold {int(engine.drift_detector.dataset_drift_share_threshold * 100)}%)")
    print(f"Dataset Drift Detected: {'[YES - CRITICAL]' if summary.dataset_drift else '[NO - STABLE]'}")

    if report.performance.rolling_metrics:
        perf = report.performance
        m = perf.rolling_metrics
        print("\n" + "-" * 82)
        print("MODEL PERFORMANCE MONITORING")
        print("-" * 82)
        print(f"Samples evaluated : {m.sample_count}")
        print(f"Baseline F1 score : {perf.baseline_f1:.4f}" if perf.baseline_f1 is not None else "Baseline F1 score : N/A")
        print(f"Rolling F1 score  : {m.f1:.4f}")
        print(f"F1 drop (delta)   : {perf.f1_drop:.4f}" if perf.f1_drop is not None else "F1 drop (delta)   : N/A")
        print(f"Precision / Recall: {m.precision:.4f} / {m.recall:.4f}")
        print(f"Accuracy          : {m.accuracy:.4f}")
        print(f"Performance Status: {'[DEGRADED]' if perf.degraded else '[OK]'}")

    trig = report.trigger
    print("\n" + "=" * 82)
    status_label = "[RETRAINING REQUIRED]" if trig.retraining_required else "[NO RETRAINING REQUIRED]"
    print(f"RETRAINING TRIGGER DECISION: {status_label} (Severity: {trig.severity.upper()})")
    print("=" * 82)
    if trig.trigger_reasons:
        print("Trigger Reasons:")
        for r in trig.trigger_reasons:
            print(f"  * {r}")
    print("Recommendations:")
    for rec in trig.recommendations:
        print(f"  -> {rec}")

    print(f"\nMonitoring report saved to: {out_path}")


if __name__ == "__main__":
    main()
