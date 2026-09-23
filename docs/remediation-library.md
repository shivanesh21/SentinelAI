# Remediation Action Library (Module 7)

A small library of safe, reversible remediation actions mapped from root-cause
categories produced by the RCA module (Day 16-17). It is the actionable side of
the pipeline: risk engine → evidence → RCA → remediation plan.

## Files

- `backend/src/remediation/model.py` — `ActionSpec` (name, description, params,
  reversible, risk), `ActionResult`, `ExecutionReport`.
- `backend/src/remediation/actions.py` — `ACTION_REGISTRY` (name → metadata:
  description, risk, reversible, runner, rollback + rollback params) and the
  `execute_spec` / `rollback_spec` dispatchers.
- `backend/src/remediation/backends.py` — pluggable backends: `MockBackend`
  (deterministic, in-memory, fully reversible — the default and the safe
  baseline) and `DockerBackend` (uses the docker SDK when importable, silently
  defers to simulation otherwise).
- `backend/src/remediation/planner.py` — `ROOT_CAUSE_ACTION_MAP` and
  `RemediationPlanner.plan_for_category(...)` mapping each root-cause category
  to an ordered action list.
- `backend/src/remediation/executor.py` — `RemediationExecutor` with safety
  modes and a JSONL audit trail (`data/remediation_audit.jsonl` by default).
- `backend/scripts/demo_remediation.py` — plan + execute + rollback walkthrough.
- `backend/tests/test_remediation.py` — 14 tests.

## Actions

| Action | Backend method | Reversible | Risk |
|---|---|---|---|
| `restart_container` | `restart_container(container)` | yes | medium |
| `clear_cache` | `clear_cache(service)` | yes | low |
| `scale_replicas` | `scale_replicas(service, count)` | yes (tracked) | low |
| `create_incident_ticket` | `create_ticket(service, category, summary)` | no | low |

`scale_replicas` records the previous replica count so rollback restores it;
tickets are intentionally not auto-reversible (human follow-up required).

## Root-cause → action map

Each category maps to ordered actions (highest value first):

- `memory_leak` → restart, ticket
- `connection_pool_exhaustion` → scale up, clear cache, ticket
- `latency_spike` → scale up, clear cache, ticket
- `network_partition` → restart, ticket
- `disk_fill` → clear cache (free disk), restart, ticket
- `code_deployment` → restart (roll back image), ticket
- `traffic_spike` → scale up, clear cache, ticket
- `unknown` → ticket only

## Safety modes (Day 19)

`resolve_mode()` resolution order: explicit argument → `REMEDIATION_MODE` env →
settings (`remediation.mode`, default `approval`).

Per-action `status` outcomes: `planned` (advisory), `ok`, `requires_approval`
(autonomous, deferred to human), `failed`, `rolled_back`, `skipped`.

| Mode | Behavior |
|---|---|
| `advisory` | never executes; every action reported `planned` |
| `approval` | dry-run by default; creates a `PendingApproval`; executes only via the explicit approve flow |
| `autonomous` | auto-executes only actions allowed by policy; the rest are deferred as `requires_approval` |

### Policy gating (`src/remediation/policy.py`)

`RemediationPolicy` decides each plan action in autonomous mode via
`decisions(plan, autonomous)`:
- `execute` — action is on `autonomous_whitelist` **and** its risk ≤
  `max_risk_autonomous`;
- `propose` — safe enough but deferred (over `max_actions_per_incident` cap, or
  non-autonomous mode);
- `skip` — not allowed at all.

Defaults from `settings.yaml` `remediation.policy`:
`autonomous_whitelist: [clear_cache]`, `max_risk_autonomous: low`,
`max_actions_per_incident: 4`. So `clear_cache` auto-runs; `scale_replicas` /
`restart_container` / tickets always require a human.

### Approval flow (`src/remediation/approval.py`)

In-memory `ApprovalStore` holds `PendingApproval` records
(`request_id`, service, category, plan specs, `approval_ttl_min`). The executor
audit writes one JSONL record per `execute()` including the policy snapshot.

## API integration

`backend/src/api/main.py`:

- `POST /api/remediate` — body: `{rows, mode?, approve?}`. Advisory → plan-only.
  Approval → creates a pending approval (returns `approval_id`, `dry_run:
  true`; legacy `approve: true` executes immediately). Autonomous → executes
  policy-allowed actions, returns deferred `requires_approval` actions plus an
  `approval_id` when anything is pending.
- `POST /api/remediate/approve` — `{approval_id}`; takes the pending approval,
  executes it (`dry_run: false`), returns the `ExecutionReport`. 404 for
  unknown/expired ids. A human are-in-the-loop gate: nothing ever runs without
  explicit approve except whitelisted low-risk actions.
- `GET /api/remediate/pending` — lists outstanding approvals.

Live smoke-tested (200 OK) on all three modes: advisory → `planned,planned`/
no approval; approval create → approve → `restart_container`+ticket `ok`; and
autonomous gating (memory_leak plan deferred to `requires_approval`, visible in
the pending list) against `MockBackend`.

## Re-run

```bash
cd backend
python scripts/demo_remediation.py
python -m unittest discover tests
```