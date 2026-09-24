# Experiment Tracking (Model Registry)

Lightweight, self-contained experiment tracking for SentinelAI — no external
server required. Every trained model run is recorded as a JSON line in a
versioned registry (audit-style, matching the rest of the codebase).

## Registry

`backend/data/experiments.jsonl` — one JSON object per experiment:

```json
{
  "experiment_id": "883cea80ead9",
  "family": "baseline",
  "model": "random_forest",
  "model_version": 1,
  "dataset_version": "ds-8b9f08589a3e",
  "dataset": {},
  "hyperparameters": { "n_estimators": 400, "max_depth": 30, ... },
  "metrics": {
    "val": { "accuracy": ..., "precision": ..., "recall": ..., "f1": ...,
             "roc_auc": ..., "pr_auc": ... },
    "test": { ...same keys... },
    "threshold": 0.7,
    "label": "failure_in_next_10min"
  },
  "model_path": ".../models/...",
  "trained_at": "2026-09-23T06:22:37+00:00",
  "status": "ok",
  "notes": {}
}
```

Location is configurable via `mlops.registry` in `config/settings.yaml`
(default `data/experiments.jsonl`).

## Versioning

- **Model version**: auto-incrementing per `family/model` (v1, v2, …), assigned
  by `ExperimentStore.next_version` at record time.
- **Dataset version**: content hash over feature-table metadata (columns,
  row/service counts, window, interval) plus the split report — `ds-<sha256-12>`.
  Reproducible: identical data config = identical dataset version.

## Module

`backend/src/mlops/experiments.py`:

- `dataset_version(feature_meta, split_report)` — content-hash dataset identity.
- `registry_from_settings(settings, base)` — resolve registry path from settings.
- `Experiment` — dataclass (family, model, hyperparameters, metrics, versions,
  path, status, notes).
- `ExperimentStore` — `record`, `read(limit/family/model)`, `get`,
  `latest(family, model)`, `best(family, model, metric, split)`, `models`,
  `compare(metric, split)` (best record per family/model, ranked).
- `write_stamp(model_dir, experiment, registry_path)` — writes `mlops.json`
  next to each saved model so artefacts are self-describing.

`best`/`compare` read `metrics[split][metric]` for `status == "ok"` records;
`compare` ranks models by the chosen metric (default `f1` on `test`).

## Recording runs

The training pipeline scripts import the tracker and record an experiment after
saving each model, stamping the model directory:

- `scripts/train_baseline.py` — logistic regression, random forest
- `scripts/train_advanced.py` — XGBoost, LightGBM
- `scripts/train_sequential.py` — LSTM classifier
- `scripts/select_model.py` — risk engine

Hyperparameters come from estimator `get_params()` (sklearn/xgb/lgbm) or
`asdict(config)` (LSTM). Metrics are reported on both `val` and `test` splits
plus the decision threshold.

Backfill for already-trained models: `scripts/track_experiments.py` (idempotent;
skips `(family, model, dataset_version)` pairs already present).

## API

- `GET /api/mlops/experiments?limit=&family=&model=` — raw registry records.
- `GET /api/mlops/compare?metric=&split=` — best per family/model, ranked.

Consumed by the dashboard Models tab (performance panel: champion model,
per-model F1/ROC-AUC/accuracy, expandable hyperparameter details).