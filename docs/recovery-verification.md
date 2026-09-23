# Recovery Verification (Module 8)

After any executed remediation action, SentinelAI re-checks the key operational
metrics against a healthy baseline and records the outcome — `SUCCESS` /
`FAILURE` plus the measured **recovery time**.

## Files

- `backend/src/recovery/verifier.py` — `RecoveryVerifier`, `MetricCheck`,
  `RecoveryReport`, and the `verify_after_remediation(...)` integration helper.
- `backend/src/recovery/__init__.py` — exports.
- `backend/scripts/demo_recovery.py` — sample before/after report walkthrough.
- `backend/tests/test_recovery.py` — 14 tests.
- `backend/config/settings.yaml` → `recovery:` block.

## Metrics

Defaults (from `settings.yaml recovery.metrics`):

| Metric | Healthy threshold |
|---|---|
| `Error_rate` | ≤ 0.02 |
| `Latency_15min_avg` | ≤ 300.0 ms |
| `Memory_growth_rate` | ≤ 0.05 |

`RecoveryVerifier.snapshot(frame, service)` computes the mean of each metric
over the given window for one service (raises `ValueError` if none of the
configured metrics are present or the service has no rows).

## Decision rule

For every monitored metric with before/after means:

- `healthy`   — `after ≤ threshold` → recovered.
- `improved`  — not healthy but `after ≤ before × (1 − min_improvement)` (≥5%
  improvement, `recovery.min_improvement`) → recovered.
- `worsened`  — `after > before × (1 + min_improvement)` while above threshold
  → not recovered.
- `unchanged` — above threshold with no meaningful change → not recovered.

**Overall `SUCCESS`** iff every checked metric is recovered; anything else is
`FAILURE`. `recovery_time_sec` is `(median after_ts − median before_ts)` when
both windows carry `timestamp` columns (floored to `recovery.cooldown_sec`,
the grace period between remediation and re-check), otherwise the cooldown.

`verify_after_remediation(execution_report, before_frame, after_frame, service)`
only produces a check when the remediation actually executed (≥1 `ok` action) —
advisory/planned suggestions are not verified.

## API integration

- `POST /api/recovery/verify` — `{service, before_rows, after_rows, action?}`:
  explicit before/after comparison; the report is logged and returned.
- `GET /api/recovery/history?limit=` — recent checks from the JSONL trail
  (`data/recovery_log.jsonl`).
- **Automatic re-check**: `POST /api/remediate/approve` (and autonomous
  executions through `/api/remediate`) run the verifier automatically when the
  caller includes `after_rows` (post-remediation telemetry collected after the
  cooldown). The before-window is captured at plan time and stored on the
  pending approval; the check is appended to the recovery log and returned as
  `recovery` in the response.

Live smoke-tested (200 OK): explicit verify on the `inventory-service` disk-fill
incident → `SUCCESS`, `recovery_time=3720s`, all three metrics `healthy`; and an
approval-flow execution of `scale_replicas`+`clear_cache`+`ticket` auto-attached
the same `SUCCESS` verification, visible in the history endpoint.

## Re-run

```bash
cd backend
python scripts/demo_recovery.py
python -m unittest discover tests
```