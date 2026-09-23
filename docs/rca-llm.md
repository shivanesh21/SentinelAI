# RCA LLM Prompt Design (Module 6)

Structured prompt template that turns an evidence JSON package (Day 15) into a
fixed-schema root-cause analysis: probable root cause, evidence list, confidence
score, recommended action.

## Files

- `backend/src/rca/schema.py` — `RCAResult` dataclass + `CAUSE_CATEGORIES`; validation enforces required keys, `confidence_score` in [0, 1], and category in the allowed set. JSON round-trip via `to_dict`/`to_json`/`from_dict`/`from_json`.
- `backend/src/rca/prompt.py` — `SYSTEM_PROMPT` (RCA agent instructions, schema contract, category whitelist, evidence-only rules) + `USER_TEMPLATE` (embeds the evidence JSON). `build_rca_prompt(evidence) -> (system, user)` accepts a dict or a JSON string; `instruction_block()` returns the schema rules alone.
- `backend/src/rca/heuristic.py` — deterministic fallback classifier mapping log keywords, top-anomalous feature names, and risk components to a category (keyword hits + feature-name matches + memory_growth component), returning the same `RCAResult` schema.
- `backend/src/rca/client.py` — `RCAClient` provider dispatch: `anthropic` / `openai` / `local` (Ollama-style `POST /api/generate`) with JSON parse into `RCAResult`; any failure (no key, network, invalid JSON) falls back to the heuristic. `resolve_provider()` reads `LLM_PROVIDER` + credentials from the environment (matches `backend/.env.example`).
- `backend/scripts/demo_rca.py` — runs 6 synthetic scenarios through prompt → result.
- `backend/tests/test_rca.py` — prompt/schema/heuristic/client tests.

## Fixed JSON schema (system prompt contract)

```json
{
  "probable_root_cause": "string",
  "root_cause_category": "memory_leak | connection_pool_exhaustion | latency_spike | network_partition | disk_fill | code_deployment | traffic_spike | unknown",
  "evidence": ["string"],
  "confidence_score": 0.95,
  "recommended_action": "string"
}
```

## Prompt shape

- **system**: RCA agent instructions + schema + category whitelist + rules
  (single JSON object only, diagnosis from provided evidence only, most
  parsimonious root cause).
- **user**: `Here is the structured evidence package for an incident:`
  followed by the evidence JSON (timestamp, service, risk_score/level,
  failure_probability, components, top_anomalous_features, metrics_deltas,
  correlated_logs) and the instruction to return the JSON result.

## Provider configuration

`LLM_PROVIDER` in the environment selects the path; a provider is only used when
its credential/endpoint is present:

| Provider | Required env | Notes |
|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` | optional `ANTHROPIC_MODEL` (default `claude-3-5-sonnet-latest`) |
| `openai` | `OPENAI_API_KEY` | optional `OPENAI_MODEL` (default `gpt-4o-mini`) |
| `local` | `LOCAL_LLM_URL` (default `http://localhost:11434`) | optional `LLM_MODEL` (default `llama3`) |
| `heuristic` | none | deterministic fallback — always available |

With no key configured (the default), `resolve_provider()` returns `heuristic`
and demos/tests run fully offline.

## Sample outputs (heuristic provider — 6 synthetic scenarios)

Run `python scripts/demo_rca.py` from `backend/` for the full prompt + result.
Each scenario returns a valid `RCAResult` JSON:

1. **DB pool exhaustion** (payment-api) → `connection_pool_exhaustion`
   confidence 0.98 — pool acquire timeout; action: tune pool limit, check
   downstream latency.
2. **memory leak** (order-service) → `memory_leak` confidence 0.98 — sustained
   heap growth + GC pressure; action: heap dump + memory profile.
3. **latency spike** (auth-service) → `latency_spike` confidence 0.98 — p99 SLO
   breach + queue buildup; action: scale horizontally, enable request shedding.
4. **network partition** (notification-worker) → `network_partition`
   confidence 0.97 — broker unreachable / network-in collapsed; action: verify
   mesh/firewall/DNS.
5. **disk fill** (inventory-service) → `disk_fill` confidence 0.98 — no space
   left on device; action: free space, rotate logs, add disk alarm.
6. **code deployment regression** (payment-api) → `code_deployment`
   confidence 0.82 — elevated 5xx after rollout; action: roll back deployment.

All categories matched the scenario ground truth; `confidence_score` stayed in
[0, 1]; `evidence` cited real components/features from the package.

## API integration

`POST /api/rca` (`backend/src/api/main.py`) takes `{"rows": [...]}`, assembles
evidence via `EvidenceAssembler` (threshold 0.40), runs `RCAClient` on the
highest-risk package, and returns `{analyzed, request_id, provider, evidence, rca}`.
Smoke-tested live (200 OK, heuristic provider).

## Guardrails (Day 17)

The LLM call is wrapped as a service with validation and fallback:

- **Schema validation** (`parse_result` + `RCAResult.from_dict`): every LLM
  response is parsed with `extract_json` (strips ``` fences, extracts the JSON
  object) and validated — required keys, non-empty `probable_root_cause`,
  `confidence_score` in [0, 1]. Unknown/typo'd `root_cause_category` is coerced
  to `unknown`, never propagated.
- **Fallback**: any malformed output (transport error, non-JSON, schema
  violation) triggers the deterministic `heuristic_classify` fallback, so RCA
  always returns a valid result.

## Audit trail (Day 17)

Every RCA call is written to a JSONL audit log
(`backend/config/settings.yaml` → `rca.audit_log`, default
`backend/data/rca_audit.jsonl`):

- `src/rca/audit.py` — `RCAEvent` (request_id, timestamp, provider, service,
  input evidence, output diagnosis, status, duration_ms, error, llm_raw) +
  `AuditLogger` (append-only `record()`, tailing `read(limit)`).
- `RCAClient` records automatically when handed an `AuditLogger`
  (`analyze(evidence, request_id, audit)`); status is `ok` or
  `fallback_heuristic` (with the error).
- `POST /api/rca` returns a `request_id`; `GET /api/rca/history?limit=N` returns
  the most recent audited calls (input evidence + diagnosis) for auditability.
- Smoke-tested live: POST → 200 with `request_id`, GET history → 200 with the
  matching audit record.

## Re-run

```bash
cd backend
python scripts/demo_rca.py
python -m unittest discover tests
```