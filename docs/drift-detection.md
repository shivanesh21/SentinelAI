# Drift Detection & Performance Monitoring (Day 22 / Module 9)

SentinelAI continuously monitors the health of upstream telemetry data and predictive machine learning models in production. When feature distributions shift or model predictive performance degrades, an automated **"retraining required"** trigger is raised to prompt model retraining before operational degradation impacts incident mitigation.

---

## 1. Architecture & Components

The drift detection and performance monitoring subsystem is located in `backend/src/mlops/`:

- `backend/src/mlops/drift.py`:
  - `calculate_psi(...)`: Population Stability Index on binned continuous features.
  - `calculate_ks_test(...)`: Two-sample Kolmogorov-Smirnov test for distribution divergence.
  - `DriftDetector`: Computes per-feature drift statistics and overall dataset drift.
  - `PerformanceMonitor`: Computes rolling classification metrics ($F_1$, precision, recall, accuracy) and flags degradation against baseline performance.
  - `RetrainingTrigger`: Evaluates drift and degradation against thresholds and produces the actionable trigger verdict.
  - `DriftMonitoringEngine`: High-level coordinator that runs checks, builds the monitoring report, and persists it.
- `backend/scripts/detect_drift.py`: CLI script for batch/scheduled drift analysis and synthetic drift simulations.
- `backend/reports/evaluations/drift_monitoring_report.json`: JSON monitoring report deliverable.
- `backend/config/settings.yaml` (`mlops.drift`): Operational thresholds and report locations.
- `backend/tests/test_drift.py`: Unit and integration test suite.
- `backend/src/api/main.py`: REST API endpoints for live status and on-demand check.

---

## 2. Statistical Methodology

### 2.1 Population Stability Index (PSI)

For each continuous feature $X$, reference values (baseline training set) are binned into $K=10$ quantiles. Given empirical proportions $P_i$ (reference) and $Q_i$ (target production window) with smoothing $\epsilon = 10^{-4}$:

$$\text{PSI} = \sum_{i=1}^{K} (Q_i - P_i) \cdot \ln\left(\frac{Q_i}{P_i}\right)$$

| PSI Value Range | Classification | Action |
|---|---|---|
| $\text{PSI} < 0.10$ | **None / Stable** | No action required. |
| $0.10 \le \text{PSI} < 0.25$ | **Moderate Drift** | Flagged as warning; monitored for trend. |
| $\text{PSI} \ge 0.25$ | **Significant Drift** | Drift detected; counts toward dataset drift. |

### 2.2 Two-Sample Kolmogorov-Smirnov (KS) Test

For continuous distributions, the non-parametric KS-test evaluates whether the target sample is drawn from the same continuous distribution as the baseline reference:

$$D = \sup_x |F_{\text{ref}}(x) - F_{\text{target}}(x)|$$

A p-value $p < \alpha$ ($\alpha = 0.05$) indicates a statistically significant difference between distributions.

### 2.3 Dataset-Level Drift Rule

Dataset-level drift (`dataset_drift = True`) is triggered if:
$$\frac{\text{count}(\text{features with significant drift})}{\text{total features}} \ge \theta_{\text{drift\_share}} \quad (\theta = 0.30)$$

---

## 3. Model Performance Monitoring

When fresh ground-truth labels are available (e.g. from incident post-mortems or sliding evaluation windows):

1. **Rolling Metrics**: Computes rolling $F_1$, precision, recall, and accuracy over recent batches ($N \ge 50$).
2. **Degradation Detection**:
   $$\Delta F_1 = F_{1,\text{baseline}} - F_{1,\text{rolling}}$$
   A model is marked as **`degraded`** if:
   - $\Delta F_1 \ge \theta_{\text{drop}}$ (default: $0.15$), OR
   - $F_{1,\text{rolling}} < \theta_{\min}$ (default: $0.60$).

---

## 4. Retraining Trigger Policy

The `RetrainingTrigger` evaluates the multi-signal monitoring report and sets `retraining_required = True` when:

1. **Dataset Drift**: $\ge 30\%$ of features exhibit significant drift ($\text{PSI} \ge 0.25$).
2. **Critical Feature Drift**: Key operational metrics (`Error_rate`, `Latency_15min_avg`, `Memory_growth_rate`) exhibit severe drift.
3. **Model Degradation**: Rolling $F_1$ drops by $\ge 0.15$ or drops below $0.60$.

### Severity Escalation

- **Normal**: Feature distributions stable, performance within bounds.
- **Warning**: Dataset or critical feature drift detected, but model accuracy is still maintained.
- **Critical**: Model performance has degraded below acceptable thresholds.

---

## 5. Configuration (`settings.yaml`)

```yaml
mlops:
  registry: data/experiments.jsonl
  drift:
    report_path: reports/evaluations/drift_monitoring_report.json
    psi:
      bins: 10
      moderate_threshold: 0.10
      significant_threshold: 0.25
    ks:
      alpha: 0.05
    dataset_drift_share_threshold: 0.30
    performance:
      f1_drop_threshold: 0.15
      min_f1_threshold: 0.60
      min_samples: 50
```

---

## 6. CLI Runner (`scripts/detect_drift.py`)

Run drift detection between baseline training split and test split:

```powershell
python scripts/detect_drift.py
```

### Options

- `--reference <path>`: Baseline dataset (default: `data/splits/train.parquet`).
- `--target <path>`: Target production dataset (default: `data/splits/test.parquet`).
- `--model-dir <path>`: Trained model directory (default: `models/baseline/random_forest`).
- `--simulate-drift`: Injects synthetic drift into latency, error rate, and resource metrics.
- `--simulate-degradation`: Perturbs model predictions to verify CRITICAL retraining escalation.
- `--output <path>`: Custom destination for the monitoring JSON report.

---

## 7. REST API Endpoints

### `GET /api/mlops/drift`
Returns the latest drift monitoring report, status, and retraining decision.

**Response Example:**
```json
{
  "timestamp": "2026-09-23T15:27:11.317174+00:00",
  "reference_samples": 8540,
  "target_samples": 2880,
  "data_drift": {
    "total_features": 26,
    "drifted_features_count": 22,
    "drift_share": 0.8462,
    "dataset_drift": true,
    "drifted_features": ["Error_rate", "Latency_15min_avg", "..."]
  },
  "performance": {
    "baseline_f1": 0.8547,
    "rolling_metrics": {
      "sample_count": 2880,
      "f1": 0.8547,
      "precision": 0.9583,
      "recall": 0.7713,
      "accuracy": 0.9701
    },
    "f1_drop": 0.0,
    "degraded": false,
    "status": "ok"
  },
  "trigger": {
    "retraining_required": true,
    "severity": "warning",
    "trigger_reasons": [
      "Dataset drift threshold breached: 22/26 features (84.6%) exhibit significant drift (threshold: 30%).",
      "Critical operational feature(s) drifted significantly: ['Error_rate', 'Latency_15min_avg']."
    ],
    "recommendations": [
      "Scheduled model retraining recommended: input feature distributions have shifted significantly from baseline.",
      "Inspect upstream service telemetry anomalies affecting: ['Error_rate', 'Latency_15min_avg'].",
      "Update feature baseline and evaluate retrained model candidates via experiment registry."
    ]
  }
}
```

### `POST /api/mlops/drift/check`
Runs real-time distribution drift checks on incoming batches of production telemetry rows.

**Request Payload:**
```json
{
  "rows": [
    {
      "CPU_5min_avg": 42.1,
      "Error_rate": 0.04,
      "Latency_15min_avg": 280.0,
      ...
    }
  ],
  "baseline_f1": 0.85
}
```
