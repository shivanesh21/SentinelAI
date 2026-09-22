from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.risk import RiskEngine
from src.telemetry.config import load_settings


def main() -> None:
    settings = load_settings(ROOT / "config" / "settings.yaml")

    # Load all individual model metrics
    baseline = json.loads((ROOT / "reports" / "evaluations" / "baseline_report.json").read_text())
    advanced = json.loads((ROOT / "reports" / "evaluations" / "advanced_report.json").read_text())
    sequential = json.loads((ROOT / "reports" / "evaluations" / "sequential_report.json").read_text())
    risk = json.loads((ROOT / "reports" / "evaluations" / "risk_report.json").read_text())

    # Build comparison table (test split)
    models = {
        "Logistic Regression": baseline["logistic_regression"]["test"]["metrics"],
        "Random Forest": baseline["random_forest"]["test"]["metrics"],
        "XGBoost": advanced["xgboost"]["test"]["metrics"],
        "LightGBM": advanced["lightgbm"]["test"]["metrics"],
        "LSTM (seq_len=10)": sequential["test"]["metrics"],
        "Risk Engine (fused)": risk["splits"]["test"]["risk_detection"],
    }

    print("=" * 100)
    print("MODEL COMPARISON — TEST SPLIT (chronological, same features, same labels)")
    print("=" * 100)
    header = f"{'Model':<28} {'F1':>7} {'Precision':>10} {'Recall':>8} {'ROC-AUC':>8} {'PR-AUC':>8} {'Accuracy':>8}"
    print(header)
    print("-" * 100)
    for name, m in models.items():
        print(f"{name:<28} {m['f1']:>7.4f} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['roc_auc']:>8.4f} {m['pr_auc']:>8.4f} {m['accuracy']:>8.4f}")

    # Validation comparison
    print("\n" + "=" * 100)
    print("MODEL COMPARISON — VAL SPLIT")
    print("=" * 100)
    val_models = {
        "Logistic Regression": baseline["logistic_regression"]["val"]["metrics"],
        "Random Forest": baseline["random_forest"]["val"]["metrics"],
        "XGBoost": advanced["xgboost"]["val"]["metrics"],
        "LightGBM": advanced["lightgbm"]["val"]["metrics"],
        "LSTM (seq_len=10)": sequential["val"]["metrics"],
    }
    for name, m in val_models.items():
        print(f"{name:<28} {m['f1']:>7.4f} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['roc_auc']:>8.4f} {m['pr_auc']:>8.4f} {m['accuracy']:>8.4f}")

    # Select best single model by F1 on val (Random Forest)
    best_single = max(val_models.items(), key=lambda x: x[1]["f1"])
    print(f"\nBest single model (val F1): {best_single[0]} (F1={best_single[1]['f1']:.4f})")

    # Risk engine is the production choice — it fuses all models + operational signals
    print("\nPRODUCTION MODEL: Risk Engine (fused)")
    print("  - Combines probabilities from LR, RF, XGBoost, LightGBM, LSTM")
    print("  - Adds operational signals (Error_rate, Latency_15min_avg, Memory_growth_rate)")
    print("  - Calibrated on training split; thresholded at WARNING=0.40")
    print(f"  - Test F1: {risk['splits']['test']['risk_detection']['f1']:.4f}, Recall: {risk['splits']['test']['risk_detection']['recall']:.4f}")
    print(f"  - Test ROC-AUC: {risk['splits']['test']['risk_detection']['roc_auc']:.4f}, PR-AUC: {risk['splits']['test']['risk_detection']['pr_auc']:.4f}")

    # Save final model selection report
    report = {
        "selection_criteria": "highest val F1 for single model; fused engine for production",
        "best_single_model": {
            "name": best_single[0],
            "val_f1": best_single[1]["f1"],
            "test_f1": models[best_single[0]]["f1"],
        },
        "production_model": "RiskEngine",
        "production_metrics": {
            "val": risk["splits"]["val"]["risk_detection"],
            "test": risk["splits"]["test"]["risk_detection"],
        },
        "individual_models_test": {name: m for name, m in models.items() if name != "Risk Engine (fused)"},
        "individual_models_val": {name: m for name, m in val_models.items()},
        "tuning_summary": {
            "lstm_best_config": sequential["config"],
            "xgboost_best_params": advanced["xgboost"]["val"].get("best_params", {}),
            "lightgbm_best_params": advanced["lightgbm"]["val"].get("best_params", {}),
        },
    }

    out_path = ROOT / "reports" / "evaluations" / "model_selection_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nModel selection report written to {out_path}")

    # Save the final production model artifact (RiskEngine with calibration)
    print("\nSaving final production model artifact...")
    engine = RiskEngine.load(ROOT / "models", settings)
    # Ensure calibration is loaded (should already be from score_risk.py)
    if not engine.references:
        train_store = __import__("src.features.store", fromlist=["FeatureStore"]).FeatureStore(ROOT / "data" / "features")
        train = train_store.load_split("train", ROOT / "data" / "splits")
        engine.calibrate(train)
    engine.save_calibration(ROOT / "models" / "risk" / "calibration.json")
    print(f"Production model calibration saved to {ROOT / 'models' / 'risk' / 'calibration.json'}")


if __name__ == "__main__":
    main()