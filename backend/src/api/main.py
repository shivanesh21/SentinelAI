from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.features.store import FeatureStore
from src.evidence import EvidenceAssembler
from src.recovery import RecoveryVerifier, verify_after_remediation
from src.remediation import (
    ApprovalStore,
    MockBackend,
    RemediationExecutor,
    RemediationPolicy,
    plan_for_category,
    policy_from_config,
)
from src.mlops import DriftMonitoringEngine
from src.risk import RiskEngine
from src.rca import AuditLogger, RCAClient, new_request_id
from src.telemetry.config import load_settings

BACKEND = Path(__file__).resolve().parents[2]
FRONTEND = BACKEND.parent / "frontend"

settings = load_settings(BACKEND / "config" / "settings.yaml")

app = FastAPI(title="SentinelAI API", version="0.5.0", description="Predictive incident detection backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_feature_store = FeatureStore(BACKEND / "data" / "features")
_split_dir = BACKEND / "data" / "splits"
_feature_table_cache = None
_risk_engine = None
_rca_service = None
_approval_store = ApprovalStore()


def _get_risk_engine() -> RiskEngine:
    global _risk_engine
    if _risk_engine is None:
        _risk_engine = RiskEngine.load(BACKEND / "models", settings)
    return _risk_engine


def _rca_audit_path():
    rel = settings.get("rca", {}).get("audit_log", "data/rca_audit.jsonl")
    return BACKEND / rel if not Path(rel).is_absolute() else Path(rel)


def _get_rca_service() -> RCAClient:
    global _rca_service
    if _rca_service is None:
        _rca_service = RCAClient(audit=AuditLogger(_rca_audit_path()))
    return _rca_service


def _remediation_policy() -> RemediationPolicy:
    config = settings.get("remediation", {})
    policy = config.get("policy", {})
    return RemediationPolicy(
        autonomous_whitelist=policy.get("autonomous_whitelist", ["clear_cache"]),
        max_risk_autonomous=policy.get("max_risk_autonomous", "low"),
        max_actions_per_incident=policy.get("max_actions_per_incident", 4),
    )


def _get_remediation_service(mode: str | None = None) -> RemediationExecutor:
    rel = settings.get("remediation", {}).get("audit_log", "data/remediation_audit.jsonl")
    audit_path = BACKEND / rel if not Path(rel).is_absolute() else Path(rel)
    return RemediationExecutor(backend=MockBackend(), mode=mode, audit_path=audit_path, policy=_remediation_policy())


def _recovery_log_path():
    rel = settings.get("recovery", {}).get("log", "data/recovery_log.jsonl")
    return BACKEND / rel if not Path(rel).is_absolute() else Path(rel)


def _get_recovery_verifier() -> RecoveryVerifier:
    cfg = settings.get("recovery", {})
    return RecoveryVerifier(
        cooldown_sec=int(cfg.get("cooldown_sec", 60)),
        metrics=cfg.get("metrics"),
        thresholds=cfg.get("healthy_threshold"),
        min_improvement=float(cfg.get("min_improvement", 0.05)),
        log_path=str(_recovery_log_path()),
    )


def _recovery_check(execution_report, before_rows, after_rows, service) -> dict | None:
    if not before_rows or not after_rows:
        return None
    frames = []
    for rows in (before_rows, after_rows):
        df = pd.DataFrame(rows)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
        frames.append(df)
    action = ", ".join(r.name for r in execution_report.results if r.status == "ok") or None
    verifier = _get_recovery_verifier()
    report = verify_after_remediation(
        execution_report, frames[0], frames[1], service, action=action, verifier=verifier
    )
    if report is None:
        return None
    verifier.log(report)
    return report.to_dict()


def _get_feature_table() -> pd.DataFrame:
    global _feature_table_cache
    if _feature_table_cache is None:
        _feature_table_cache = _feature_store.read_table()
    return _feature_table_cache


def _current_frame(minutes: int = 20) -> pd.DataFrame:
    table = _get_feature_table()
    cutoff = table["timestamp"].max() - pd.Timedelta(minutes=minutes)
    return table[table["timestamp"] >= cutoff]


def _jsonable_rows(scored: pd.DataFrame) -> list[dict]:
    rows = []
    for record in scored.to_dict("records"):
        rows.append(
            {
                "timestamp": record["timestamp"].isoformat() if hasattr(record["timestamp"], "isoformat") else str(record["timestamp"]),
                "service": record["service"],
                "risk_score": round(float(record["risk_score"]), 4),
                "risk_level": record["risk_level"],
                "risk": int(record["risk"]),
                "failure_probability": round(float(record["failure_probability"]), 4),
                **{
                    name: round(float(record[name]), 4)
                    for name in ("error_rate", "latency_trend", "memory_growth", "anomaly_score")
                    if name in record and pd.notna(record[name])
                },
            }
        )
    return rows


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "sentinelai-backend",
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/overview")
def overview() -> dict:
    meta_path = BACKEND / "data" / "features" / "feature_table.meta.json"
    if not meta_path.exists():
        return {"available": False}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    split_report = {}
    report_path = _split_dir / "split_report.json"
    if report_path.exists():
        split_report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "available": True,
        "name": meta.get("name", "feature_table"),
        "n_rows": meta.get("n_rows", 0),
        "n_services": meta.get("n_services", 0),
        "services": meta.get("services", []),
        "n_features": len(meta.get("feature_columns", [])),
        "n_labels": len(meta.get("label_columns", [])),
        "prediction_window_min": meta.get("prediction_window_min"),
        "interval_sec": meta.get("interval_sec"),
        "start_ts": meta.get("start_ts"),
        "end_ts": meta.get("end_ts"),
        "splits": split_report,
    }


@app.get("/api/risk/current")
def risk_current(minutes: int = 20) -> dict:
    try:
        engine = _get_risk_engine()
        frame = _current_frame(minutes=minutes)
        if frame.empty:
            return {"as_of": None, "services": []}
        latest = engine.latest_by_service(frame)
        latest = _jsonable_rows(latest)
        ranking = [int(x["risk"]) for x in latest]
        return {
            "as_of": latest[0]["timestamp"] if latest else None,
            "window_min": minutes,
            "services": latest,
            "n_flagged": sum(ranking),
        }
    except Exception as exc:  # pragma: no cover - defensive for the demo endpoint
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/risk/score")
def risk_score(payload: dict) -> dict:
    rows = payload.get("rows")
    if not rows:
        raise HTTPException(status_code=400, detail="body must include a non-empty 'rows' list")
    try:
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
        required = {c for c in _get_risk_engine().feature_columns}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        engine = _get_risk_engine()
        latest = _jsonable_rows(engine.latest_by_service(df))
        return {"services": latest, "n_flagged": sum(int(x["risk"]) for x in latest)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/rca")
def rca_analyze(payload: dict) -> dict:
    """Assemble evidence for the given rows, then run RCA on the highest-risk
    service. LLM output is schema-validated with a heuristic fallback on
    malformed output; every call is written to the audit trail."""
    rows = payload.get("rows")
    if not rows:
        raise HTTPException(status_code=400, detail="body must include a non-empty 'rows' list")
    request_id = payload.get("request_id") or new_request_id()
    try:
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
        engine = _get_risk_engine()
        missing = set(engine.feature_columns) - set(df.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")

        assembler = EvidenceAssembler(engine, threshold=0.40, feature_columns=engine.feature_columns)
        packages = assembler.assemble(df)
        if not packages:
            return {"analyzed": False, "request_id": request_id, "reason": "no risk crossing the threshold"}

        package = packages[0]
        client = _get_rca_service()
        finding = client.analyze(package.to_dict(), request_id=request_id)

        audit = client.audit
        trail_id = audit.read(1)[-1]["request_id"] if audit is not None else request_id
        return {
            "analyzed": True,
            "request_id": trail_id,
            "provider": client.provider,
            "evidence": package.to_dict(),
            "rca": finding.to_dict(),
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/rca/history")
def rca_history(limit: int = 50) -> dict:
    """Return the most recent audited RCA calls (input evidence + diagnosis)."""
    limit = max(1, min(limit, 500))
    audit = AuditLogger(_rca_audit_path())
    return {"records": audit.read(limit)}


def _serialize_plan(plan) -> list[dict]:
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "params": spec.params,
            "reversible": spec.reversible,
            "risk": spec.risk,
        }
        for spec in plan
    ]


@app.post("/api/remediate")
def remediate(payload: dict) -> dict:
    """End-to-end: risk engine -> evidence -> RCA -> remediation plan, governed
    by the safety mode:
      advisory    : recommendation only (never executes)
      approval    : returns a pending approval id for human sign-off, unless
                    approve=true is passed (legacy immediate execution)
      autonomous  : policy-validated execution of whitelisted safe actions;
                    deferred actions get a pending approval id.
    Every execution is written to the remediation audit trail."""
    rows = payload.get("rows")
    if not rows:
        raise HTTPException(status_code=400, detail="body must include a non-empty 'rows' list")
    request_id = payload.get("request_id") or new_request_id()
    mode = payload.get("mode")
    approve = bool(payload.get("approve", False))
    try:
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
        engine = _get_risk_engine()
        missing = set(engine.feature_columns) - set(df.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")

        packages = EvidenceAssembler(engine, threshold=0.40, feature_columns=engine.feature_columns).assemble(df)
        if not packages:
            return {"analyzed": False, "request_id": request_id, "reason": "no risk crossing the threshold"}

        package = packages[0]
        client = _get_rca_service()
        finding = client.analyze(package.to_dict(), request_id=request_id)

        plan = plan_for_category(
            finding.root_cause_category, package.service, context={"current_replicas": 1}
        )
        executor = _get_remediation_service(mode=mode)
        report = executor.execute(plan, approve=approve and executor.mode == "approval")
        policy = _remediation_policy()

        approval_id = None
        if executor.mode == "approval" and not approve:
            approval_id = _approval_store.create(
                request_id, package.service, finding.root_cause_category, plan,
                context={"before_rows": rows},
            )
        elif executor.mode == "autonomous" and report.pending:
            deferred = [
                spec for (spec, decision, _) in policy.decisions(plan, autonomous=True) if decision == "propose"
            ]
            approval_id = _approval_store.create(
                request_id, package.service, finding.root_cause_category, deferred,
                context={"before_rows": rows},
            )

        recovery = _recovery_check(report, rows if report.executed else None, payload.get("after_rows"), package.service)

        return {
            "analyzed": True,
            "request_id": request_id,
            "category": finding.root_cause_category,
            "mode": report.mode,
            "dry_run": report.dry_run,
            "policy": policy.to_dict(),
            "approval_id": approval_id,
            "rca": finding.to_dict(),
            "plan": _serialize_plan(plan),
            "report": report.to_dict(),
            "recovery": recovery,
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/remediate/approve")
def remediate_approve(payload: dict) -> dict:
    """Human sign-off: execute the plan stored under an approval id."""
    approval_id = payload.get("approval_id")
    if not approval_id:
        raise HTTPException(status_code=400, detail="body must include 'approval_id'")
    pending = _approval_store.take(approval_id)
    if pending is None:
        raise HTTPException(status_code=404, detail=f"unknown approval_id: {approval_id}")
    try:
        executor = _get_remediation_service(mode="approval")
        report = executor.execute(pending.to_specs(), approve=True)
        recovery = _recovery_check(
            report,
            pending.context.get("before_rows") if report.executed else None,
            payload.get("after_rows"),
            pending.service,
        )
        return {
            "approved": True,
            "approval_id": approval_id,
            "request_id": pending.request_id,
            "category": pending.category,
            "mode": report.mode,
            "dry_run": report.dry_run,
            "plan": _serialize_plan(pending.to_specs()),
            "report": report.to_dict(),
            "recovery": recovery,
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/remediate/pending")
def remediation_pending() -> dict:
    """List pending approvals awaiting human sign-off."""
    return {"pending": [p.to_dict() for p in _approval_store.list()]}


@app.post("/api/recovery/verify")
def recovery_verify(payload: dict) -> dict:
    """Explicit before/after recovery check: compares the given metric windows,
    logs SUCCESS/FAILURE + recovery time to the recovery audit trail."""
    service = payload.get("service")
    before_rows = payload.get("before_rows")
    after_rows = payload.get("after_rows")
    if not service:
        raise HTTPException(status_code=400, detail="body must include 'service'")
    if not before_rows or not after_rows:
        raise HTTPException(status_code=400, detail="body must include 'before_rows' and 'after_rows'")
    try:
        frames = []
        for rows in (before_rows, after_rows):
            df = pd.DataFrame(rows)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
            frames.append(df)
        verifier = _get_recovery_verifier()
        report = verifier.verify(frames[0], frames[1], service, action=payload.get("action"))
        verifier.log(report)
        return report.to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/recovery/history")
def recovery_history(limit: int = 20) -> dict:
    """Recent recovery verifications (SUCCESS/FAILURE + recovery time)."""
    limit = max(1, min(limit, 200))
    return {"records": _get_recovery_verifier().read(limit)}


_drift_engine = None


def _get_drift_engine() -> DriftMonitoringEngine:
    global _drift_engine
    if _drift_engine is None:
        _drift_engine = DriftMonitoringEngine(settings=settings, base_dir=BACKEND)
    return _drift_engine


@app.get("/api/mlops/drift")
def mlops_drift_status() -> dict:
    """Return the latest feature drift monitoring report, model degradation status,
    and retraining requirement decision."""
    report_rel = settings.get("mlops", {}).get("drift", {}).get("report_path", "reports/evaluations/drift_monitoring_report.json")
    report_path = BACKEND / report_rel if not Path(report_rel).is_absolute() else Path(report_rel)
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))

    ref_path = BACKEND / "data" / "splits" / "train.parquet"
    target_path = BACKEND / "data" / "splits" / "test.parquet"
    if not ref_path.exists() or not target_path.exists():
        raise HTTPException(status_code=404, detail="baseline and test datasets not found to generate drift report")

    engine = _get_drift_engine()
    ref_df = pd.read_parquet(ref_path)
    target_df = pd.read_parquet(target_path)

    y_true = target_df["failure_in_next_10min"].values if "failure_in_next_10min" in target_df.columns else None
    report = engine.run(ref_df, target_df, y_true=y_true, y_pred=y_true, baseline_f1=0.85)
    engine.save_report(report, report_path)
    return report.to_dict()


@app.post("/api/mlops/drift/check")
def mlops_drift_check(payload: dict) -> dict:
    """Run real-time drift check on submitted target telemetry rows against the
    baseline training distribution."""
    rows = payload.get("rows")
    if not rows:
        raise HTTPException(status_code=400, detail="body must include a non-empty 'rows' list")

    try:
        target_df = pd.DataFrame(rows)
        ref_path = BACKEND / "data" / "splits" / "train.parquet"
        if ref_path.exists():
            ref_df = pd.read_parquet(ref_path)
        else:
            ref_df = _get_feature_table()

        engine = _get_drift_engine()
        y_true = target_df["failure_in_next_10min"].values if "failure_in_next_10min" in target_df.columns else None
        y_pred = payload.get("predictions")
        baseline_f1 = payload.get("baseline_f1", 0.85)

        report = engine.run(
            reference_df=ref_df,
            target_df=target_df,
            y_true=y_true,
            y_pred=y_pred,
            baseline_f1=baseline_f1,
            metadata={"source": "api_check"},
        )
        return report.to_dict()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")