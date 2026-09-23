from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mlops import Experiment, ExperimentStore, dataset_version, registry_from_settings
from src.telemetry.config import load_settings


def load_report(name: str) -> dict:
    path = ROOT / "reports" / "evaluations" / name
    if not path.exists():
        print(f"  (missing report: {name})")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def sklearn_params(model_dir: Path) -> dict:
    try:
        import joblib

        pipe = joblib.load(model_dir / "model.joblib")
        estimator = pipe.named_steps.get("clf") if hasattr(pipe, "named_steps") else pipe
        return estimator.get_params()
    except Exception:
        return {}


def build_experiments(settings: dict) -> list[dict]:
    baseline = load_report("baseline_report.json")
    advanced = load_report("advanced_report.json")
    sequential = load_report("sequential_report.json")
    risk = load_report("risk_report.json")

    experiments = []

    for name, data in baseline.items():
        if "val" not in data:
            continue
        model_dir = ROOT / "models" / "baseline" / name
        experiments.append(
            {
                "family": "baseline",
                "model": name,
                "hyperparameters": sklearn_params(model_dir),
                "metrics": {
                    "val": data["val"].get("metrics", {}),
                    "val_threshold": float(data["val"].get("threshold", 0.5)),
                    "test": data["test"].get("metrics", {}),
                    "threshold": float(data["test"].get("threshold", 0.5)),
                },
                "model_path": str(model_dir),
                "notes": {"top_features": (data["val"].get("feature_importance") or [])[:5]},
            }
        )

    for name, data in advanced.items():
        if "val" not in data:
            continue
        model_dir = ROOT / "models" / "advanced" / name
        experiments.append(
            {
                "family": "advanced",
                "model": name,
                "hyperparameters": sklearn_params(model_dir),
                "metrics": {
                    "val": data["val"].get("metrics", {}),
                    "val_threshold": float(data["val"].get("threshold", 0.5)),
                    "test": data["test"].get("metrics", {}),
                    "threshold": float(data["test"].get("threshold", 0.5)),
                },
                "model_path": str(model_dir),
                "notes": {
                    "tune": bool(data["val"].get("best_params")),
                    "best_params": data["val"].get("best_params") or {},
                    "top_features": (data["val"].get("feature_importance") or [])[:5],
                },
            }
        )

    if sequential.get("test"):
        model_dir = ROOT / "models" / "prediction" / "lstm_classifier"
        experiments.append(
            {
                "family": "sequential",
                "model": "lstm_classifier",
                "hyperparameters": {
                    **{k: v for k, v in sequential.get("config", {}).items()},
                    "epochs_run": sequential.get("training", {}).get("epochs_run"),
                },
                "metrics": {
                    "val": sequential["val"].get("metrics", {}),
                    "val_threshold": float(sequential["val"].get("threshold", 0.5)),
                    "test": sequential["test"].get("metrics", {}),
                    "threshold": float(sequential["test"].get("threshold", 0.5)),
                },
                "model_path": str(model_dir),
                "notes": {
                    "tune": bool(sequential.get("tuning", {}).get("enabled")),
                    "epochs_run": sequential.get("training", {}).get("epochs_run"),
                },
            }
        )

    if risk.get("splits"):
        experiments.append(
            {
                "family": "risk",
                "model": "risk_engine",
                "hyperparameters": {
                    "weights": settings["risk"]["weights"],
                    "signals": settings["risk"]["signal_map"],
                    "severity_thresholds": settings["risk"]["severity_thresholds"],
                },
                "metrics": {
                    "val": risk["splits"]["val"]["risk_detection"],
                    "test": risk["splits"]["test"]["risk_detection"],
                },
                "model_path": str(ROOT / "models" / "risk"),
                "notes": {},
            }
        )

    return experiments


def main() -> None:
    settings = load_settings(ROOT / "config" / "settings.yaml")
    registry = ExperimentStore(registry_from_settings(settings, ROOT))

    split_report = json.loads((ROOT / "data" / "splits" / "split_report.json").read_text(encoding="utf-8"))
    feature_meta = json.loads((ROOT / "data" / "features" / "feature_table.meta.json").read_text(encoding="utf-8"))
    version = dataset_version(feature_meta, split_report)

    print("=" * 78)
    print("EXPERIMENT TRACKING — Phase 4 training pipeline (Module 9)")
    print("=" * 78)
    print(f"Dataset version: {version}")
    print(f"Registry: {registry.registry_path}")
    print(f"Split counts: {split_report['counts']}")
    print(f"Existing records: {len(registry.read())}")

    for args in build_experiments(settings):
        existing = [
            r
            for r in registry.read(family=args["family"], model=args["model"])
            if r["dataset_version"] == version
        ]
        if existing:
            print(f"  [skip] {args['family']:11s} {args['model']:18s} already tracked (v{existing[-1]['model_version']})")
            continue
        experiment = Experiment(
            family=args["family"],
            model=args["model"],
            hyperparameters=args["hyperparameters"],
            metrics=args["metrics"],
            dataset_version=version,
            model_version=0,
            model_path=args["model_path"],
            notes=args["notes"],
        )
        registry.record(experiment)
        ex = registry.get(experiment.experiment_id)
        test = ex["metrics"].get("test", {})
        print(
            f"  [track] {args['family']:11s} {args['model']:18s} v{ex['model_version']}  "
            f"test_f1={test.get('f1'):.4f} roc_auc={test.get('roc_auc'):.4f}"
        )

    print("\n" + "=" * 78)
    print("MODEL COMPARISON (best test F1 per model, from the tracker)")
    print("=" * 78)
    header = f"{'family':<11s} {'model':<18s} {'ver':>3s} {'f1':>7s} {'precision':>10s} {'recall':>8s} {'roc_auc':>8s} {'dataset':>16s}"
    print(header)
    print("-" * len(header))
    for row in registry.compare(metric="f1", split="test"):
        m = row["metrics"]
        print(
            f"{row['family']:<11s} {row['model']:<18s} {row['model_version']:>3d} "
            f"{m.get('f1', 0):>7.4f} {m.get('precision', 0):>10.4f} {m.get('recall', 0):>8.4f} "
            f"{m.get('roc_auc', 0):>8.4f} {row['dataset_version']:>16s}"
        )

    print(f"\nRegistry written to {registry.registry_path}")


if __name__ == "__main__":
    main()