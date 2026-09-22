from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.features.store import FeatureStore
from src.models.sequential import (
    SequentialLSTMClassifier,
    build_sequences,
    build_sequences_for_splits,
    sequential_configs_from_settings,
    tune_sequential,
)
from src.telemetry.config import load_settings


def metric_row(entry: dict) -> dict:
    m = entry["metrics"]
    return {
        "f1": round(m["f1"], 4),
        "precision": round(m["precision"], 4),
        "recall": round(m["recall"], 4),
        "roc_auc": round(m["roc_auc"], 4),
        "pr_auc": round(m["pr_auc"], 4),
        "accuracy": round(m["accuracy"], 4),
    }


def load_tree_test_results() -> dict:
    comparison = {}
    for report_name in ("baseline_report.json", "advanced_report.json"):
        path = ROOT / "reports" / "evaluations" / report_name
        if not path.exists():
            continue
        for model_name, model_data in json.loads(path.read_text(encoding="utf-8")).items():
            if "test" in model_data:
                comparison[model_name] = metric_row(model_data["test"])
    return comparison


def print_comparison(comparison: dict, lstm_test: dict) -> None:
    header = ["model", "f1", "precision", "recall", "roc_auc", "pr_auc"]
    print("\n=== Test-split comparison ===")
    print(f"{'model':<24}{'f1':>10}{'precision':>11}{'recall':>9}{'roc_auc':>9}{'pr_auc':>9}")
    for name, row in comparison.items():
        print(
            f"{name:<24}{row['f1']:>10}{row['precision']:>11}{row['recall']:>9}{row['roc_auc']:>9}{row['pr_auc']:>9}"
        )
    print(
        f"{'lstm_classifier':<24}{lstm_test['f1']:>10}{lstm_test['precision']:>11}{lstm_test['recall']:>9}{lstm_test['roc_auc']:>9}{lstm_test['pr_auc']:>9}"
    )


def main():
    settings = load_settings(ROOT / "config" / "settings.yaml")
    label_col = "failure_in_next_10min"

    store = FeatureStore(ROOT / "data" / "features")
    feature_columns = store.load_table_meta()["feature_columns"]
    splits = {name: store.load_split(name, ROOT / "data" / "splits") for name in ("train", "val", "test")}

    config, tune, tune_space = sequential_configs_from_settings(settings)
    print(f"Sequence config: {config} (tune={tune})")
    seq = build_sequences_for_splits(splits, feature_columns, label_col, config.seq_len)
    X, y, meta = seq["train"]
    print(f"Train windows: {X.shape}, positives {int(y.sum())} ({y.mean():.4f})")
    for name in ("val", "test"):
        nX, ny, nmeta = seq[name]
        print(f"{name} windows: {nX.shape}, positives {int(ny.sum())} ({ny.mean():.4f})")

    tuning_results = None
    if tune and tune_space:
        print("Tuning LSTM hyperparameters on validation F1 ...")
        config, tuning_results = tune_sequential(splits, feature_columns, tune_space, label_col)
        print(f"Best config: {config}")
        seq = build_sequences_for_splits(splits, feature_columns, label_col, config.seq_len)
        X, y, meta = seq["train"]

    model = SequentialLSTMClassifier(config).fit(X, y, seq["val"][0], seq["val"][1])

    epochs_run = len(model.history_.get("auc", []))
    last = {k: round(model.history_[k][-1], 4) for k in ("loss", "auc") if model.history_.get(k)}
    print(f"Trained {epochs_run} epochs (early-stop patience {config.patience}); last {last}")
    print(f"Class weight: {model.class_weight_}")

    val_eval = model.evaluate(seq["val"][0], seq["val"][1])
    test_eval = model.evaluate(seq["test"][0], seq["test"][1], threshold=val_eval["threshold"])
    print(f"Val metrics: {val_eval}")
    print(f"Test metrics: {test_eval}")

    model_dir = ROOT / "models" / "prediction" / "lstm_classifier"
    model.save(model_dir)
    print(f"Saved to {model_dir}")

    comparison = load_tree_test_results()
    lstm_test = metric_row(test_eval)
    print_comparison(comparison, lstm_test)

    report = {
        "model": "lstm_classifier",
        "config": {
            "seq_len": config.seq_len,
            "lstm_units": config.lstm_units,
            "dropout": config.dropout,
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "patience": config.patience,
            "seed": config.seed,
        },
        "tuning": {"enabled": bool(tuning_results), "results": tuning_results or []},
        "windows": {name: {"n": int(seq[name][0].shape[0]), "positives": int(seq[name][1].sum())} for name in seq},
        "training": {
            "epochs_run": epochs_run,
            "class_weight": model.class_weight_,
            "history": {k: model.history_[k] for k in ("loss", "auc", "val_auc") if k in model.history_},
        },
        "val": {"threshold": val_eval["threshold"], "metrics": val_eval["metrics"]},
        "test": {"threshold": test_eval["threshold"], "metrics": test_eval["metrics"]},
        "comparison_test_split": {**comparison, "lstm_classifier": lstm_test},
    }
    report_path = ROOT / "reports" / "evaluations" / "sequential_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()