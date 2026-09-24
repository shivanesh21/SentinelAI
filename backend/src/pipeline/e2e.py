from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.anomaly import IsolationForestDetector
from src.evidence import EvidenceAssembler
from src.features import build_features, feature_config_from_settings
from src.features.store import FeatureStore
from src.recovery import RecoveryVerifier
from src.remediation import (
    ApprovalStore,
    MockBackend,
    RemediationExecutor,
    plan_for_category,
    policy_from_config,
)
from src.rca import AuditLogger, RCAClient, new_request_id
from src.risk import RiskEngine
from src.telemetry.config import load_scenarios, load_settings, load_services
from src.telemetry.simulator.failure_scenarios import ScenarioSpec
from src.telemetry.simulator.generator import TelemetryGenerator
from src.telemetry.simulator.service import ServiceSimulator

DEFAULT_THRESHOLD = 0.40
SCENARIO_TYPES = [
    "memory_leak",
    "connection_pool_exhaustion",
    "latency_spike",
    "network_partition",
    "disk_fill",
]


class AnomalyAdapter:
    """Bridge a fitted anomaly detector into the RiskEngine contract.

    RiskEngine expects ``score(df)`` to return a frame carrying a
    ``{primary}_score`` column and a ``primary`` attribute; BaseDetector
    returns a raw score array. This adapter closes that seam."""

    def __init__(self, detector: object) -> None:
        self.detector = detector
        self.primary = getattr(detector, "name", "anomaly")

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        scores = np.asarray(self.detector.score(df), dtype=float)
        out = df[["timestamp", "service"]].copy()
        out[f"{self.primary}_score"] = scores
        return out


def _to_frame(rows: Any) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        return rows.copy()
    frame = pd.DataFrame(list(rows or []))
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="mixed")
    return frame


def latest_per_service(scored: pd.DataFrame, group_col: str = "service") -> pd.DataFrame:
    frame = scored.sort_values("timestamp")
    latest = frame.drop_duplicates(subset=[group_col], keep="last").copy()
    return latest.sort_values("risk_score", ascending=False).reset_index(drop=True)


def _serialize_plan(plan) -> list[dict[str, Any]]:
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


def _latest_rows(latest: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for record in latest.to_dict("records"):
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
                    if name in record and not pd.isna(record[name])
                },
            }
        )
    return rows


class PipelineEngine:
    """Day 24 end-to-end orchestrator.

    Chains Telemetry -> Features -> Anomaly + Failure Prediction -> Risk ->
    RCA -> Remediation -> Recovery, mirroring the FastAPI endpoints so an
    in-process run matches what /api/pipeline/run executes."""

    def __init__(
        self,
        root: str | Path,
        settings: dict | None = None,
        mode: str | None = None,
        approval_store: ApprovalStore | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        risk_engine: RiskEngine | None = None,
    ) -> None:
        self.root = Path(root)
        self.settings = settings if settings is not None else load_settings(self.root / "config" / "settings.yaml")
        self.mode = mode
        self.threshold = threshold
        self.approval_store = approval_store or ApprovalStore()
        self._risk_engine_override = risk_engine
        self._engine: RiskEngine | None = None
        self._detector: AnomalyAdapter | None = None
        self._rca: RCAClient | None = None
        self._verifier: RecoveryVerifier | None = None
        self._feature_config = feature_config_from_settings(self.settings)

    # -------------------------------------------------------------- components

    def _train_split(self) -> pd.DataFrame:
        store = FeatureStore(self.root / "data" / "features")
        return store.load_split("train", self.root / "data" / "splits")

    def risk_engine(self) -> RiskEngine:
        if self._engine is None:
            if self._risk_engine_override is not None:
                engine = self._risk_engine_override
            else:
                engine = RiskEngine.load(self.root / "models", self.settings)
            if not engine.references:
                engine.calibrate(self._train_split())
            if self._detector is None:
                self._detector = self._anomaly_adapter(engine.feature_columns)
            engine.anomaly_detector = self._detector
            self._engine = engine
        return self._engine

    def _anomaly_adapter(self, feature_columns: list[str]) -> AnomalyAdapter:
        cfg = self.settings.get("anomaly", {}) or {}
        forest = cfg.get("isolation_forest", {}) or {}
        contamination = float(cfg.get("contamination", 0.05))
        train = self._train_split()
        detector = IsolationForestDetector(
            contamination=contamination,
            n_estimators=int(forest.get("n_estimators", 200)),
            random_state=int(forest.get("random_state", 42)),
        )
        detector.fit(train, feature_columns)
        detector.fit_threshold(detector.score(train))
        return AnomalyAdapter(detector)

    def anomaly_detector(self) -> AnomalyAdapter:
        if self._detector is None:
            self.risk_engine()
        return self._detector

    def rca_client(self) -> RCAClient:
        if self._rca is None:
            rel = self.settings.get("rca", {}).get("audit_log", "data/rca_audit.jsonl")
            path = self.root / rel if not Path(rel).is_absolute() else Path(rel)
            self._rca = RCAClient(audit=AuditLogger(path))
        return self._rca

    def recovery_verifier(self) -> RecoveryVerifier:
        if self._verifier is None:
            cfg = self.settings.get("recovery", {}) or {}
            rel = cfg.get("log", "data/recovery_log.jsonl")
            log_path = self.root / rel if not Path(rel).is_absolute() else Path(rel)
            self._verifier = RecoveryVerifier(
                cooldown_sec=int(cfg.get("cooldown_sec", 60)),
                metrics=cfg.get("metrics"),
                thresholds=cfg.get("healthy_threshold"),
                min_improvement=float(cfg.get("min_improvement", 0.05)),
                log_path=str(log_path),
            )
        return self._verifier

    def _executor(self, mode: str | None) -> RemediationExecutor:
        rel = self.settings.get("remediation", {}).get("audit_log", "data/remediation_audit.jsonl")
        audit_path = self.root / rel if not Path(rel).is_absolute() else Path(rel)
        return RemediationExecutor(
            backend=MockBackend(),
            mode=mode,
            audit_path=audit_path,
            policy=policy_from_config(self.settings),
        )

    # ------------------------------------------------------------------ stages

    def readiness(self) -> dict:
        checks: list[dict[str, Any]] = []
        for name, ok, detail in (
            ("telemetry", True, "in-process simulated generator"),
            ("features", self.root.joinpath("data/features/feature_table.meta.json").exists(), "feature engineering configured"),
            ("models", self.root.joinpath("models/risk/calibration.json").exists(), "trained classifiers + risk calibration"),
            ("anomaly", True, "isolation forest fitted on the train split at first use"),
            ("rca", True, "heuristic fallback (no LLM keys configured)"),
            ("remediation", True, "action library + policy + mock backend"),
            ("recovery", True, "metric verifier + JSONL audit"),
            ("mlops", self.root.joinpath("data/experiments.jsonl").exists(), "experiment registry"),
            ("dashboard", self.root.parent.joinpath("frontend").exists(), "static frontend served at /"),
        ):
            checks.append({"stage": name, "ready": bool(ok), "detail": detail})
        mode = self.mode or self.settings.get("remediation", {}).get("mode", "approval")
        return {"ready": all(c["ready"] for c in checks), "mode": mode, "stages": checks}

    def features_from_raw(self, rows: Any) -> pd.DataFrame:
        return build_features(_to_frame(rows), self._feature_config)

    def assemble(self, features: pd.DataFrame) -> list:
        return EvidenceAssembler(
            self.risk_engine(), threshold=self.threshold, feature_columns=self.risk_engine().feature_columns
        ).assemble(features)

    def analyse(self, rows: Any) -> dict:
        raw = _to_frame(rows)
        features = self.features_from_raw(raw)
        scored = self.risk_engine().score(features)
        latest = latest_per_service(scored)
        packages = self.assemble(features)
        return {
            "raw_rows": int(len(raw)),
            "services": sorted(raw["service"].unique().tolist()) if "service" in raw.columns else [],
            "n_features": int(len(self.risk_engine().feature_columns)),
            "detection": {
                "rows": int(len(scored)),
                "n_services": int(latest.shape[0]),
                "n_flagged": int(latest["risk"].sum()),
                "latest": latest,
            },
            "evidence": packages,
        }

    def remediate(self, analysed: dict, mode: str | None = None, per_service: bool = False) -> dict:
        mode = mode or self.mode or self.settings.get("remediation", {}).get("mode", "approval")
        packages = analysed["evidence"]
        if not packages:
            return {
                "analyzed": False,
                "mode": mode,
                "reason": "no risk crossing the threshold",
                "services": analysed["services"],
            }
        ordered = sorted(packages, key=lambda p: p.risk_score, reverse=True)
        chosen: list = []
        if per_service:
            seen = set()
            for package in ordered:
                if package.service not in seen:
                    seen.add(package.service)
                    chosen.append(package)
        else:
            chosen = [ordered[0]]

        client = self.rca_client()
        policy = policy_from_config(self.settings)
        outcomes = []
        for package in chosen:
            request_id = new_request_id()
            finding = client.analyze(package.to_dict(), request_id=request_id)
            plan = plan_for_category(finding.root_cause_category, package.service, context={"current_replicas": 1})
            executor = self._executor(mode)
            report = executor.execute(plan, approve=False)
            approval_id = None
            if executor.mode == "approval":
                approval_id = self.approval_store.create(
                    request_id, package.service, finding.root_cause_category, plan,
                    context={"before_rows": package.to_dict().get("metrics_deltas", {})},
                )
            elif executor.mode == "autonomous" and report.pending:
                deferred = [
                    spec
                    for (spec, decision, _) in policy.decisions(plan, autonomous=True)
                    if decision == "propose"
                ]
                approval_id = self.approval_store.create(
                    request_id, package.service, finding.root_cause_category, deferred,
                    context={"before_rows": package.to_dict().get("metrics_deltas", {})},
                )
            outcomes.append(
                {
                    "service": package.service,
                    "risk_score": package.risk_score,
                    "risk_level": package.risk_level,
                    "request_id": request_id,
                    "category": finding.root_cause_category,
                    "provider": client.provider,
                    "rca": finding.to_dict(),
                    "plan": _serialize_plan(plan),
                    "report": report.to_dict(),
                    "approval_id": approval_id,
                    "mode": report.mode,
                    "dry_run": report.dry_run,
                    "executed": report.executed,
                }
            )
        return {
            "analyzed": True,
            "mode": mode,
            "executed": sum(1 for o in outcomes if o["executed"] > 0),
            "outcomes": outcomes,
        }

    def recover(self, before_features: pd.DataFrame, after_features: pd.DataFrame, service: str, action: str | None) -> dict | None:
        verifier = self.recovery_verifier()
        report = verifier.verify(before_features, after_features, service, action=action)
        verifier.log(report)
        return report.to_dict()

    def run(
        self,
        rows: Any,
        *,
        mode: str | None = None,
        per_service: bool = False,
        recovery: tuple[Any, Any] | None = None,
        service: str | None = None,
    ) -> dict:
        started = time.perf_counter()
        timings: dict[str, float] = {}
        resolved_mode = mode or self.mode or self.settings.get("remediation", {}).get("mode", "approval")
        engine = self.risk_engine()

        t0 = time.perf_counter()
        raw = _to_frame(rows)
        features = self.features_from_raw(raw)
        scored = engine.score(features)
        latest = latest_per_service(scored)
        timings["features_then_scoring"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        packages = self.assemble(features)
        timings["evidence"] = (time.perf_counter() - t0) * 1000.0

        flagged = int(latest["risk"].sum())
        remediation = None
        if flagged:
            analysed = {
                "services": sorted(raw["service"].unique().tolist()) if "service" in raw.columns else [],
                "evidence": packages,
            }
            t0 = time.perf_counter()
            remediation = self.remediate(analysed, mode=resolved_mode, per_service=per_service)
            timings["remediation"] = (time.perf_counter() - t0) * 1000.0

        recovery_out = None
        if recovery is not None:
            before_raw, after_raw = recovery
            t0 = time.perf_counter()
            before_features = self.features_from_raw(before_raw)
            after_features = self.features_from_raw(after_raw)
            action = None
            if remediation and remediation.get("executed"):
                action = ", ".join(
                    entry["name"]
                    for outcome in remediation["outcomes"]
                    for entry in outcome["report"].get("actions", [])
                    if entry.get("status") == "ok"
                )
            targets = []
            if service:
                targets = [service]
            elif remediation and remediation.get("outcomes"):
                targets = [o["service"] for o in remediation["outcomes"]]
            elif flagged:
                targets = [latest.iloc[0]["service"]]
            checks = []
            for svc in sorted(set(targets)):
                record = self.recover(before_features, after_features, svc, action=action)
                if record is not None:
                    checks.append(record)
            recovery_out = {"checks": checks}
            timings["recovery"] = (time.perf_counter() - t0) * 1000.0

        timings["total"] = (time.perf_counter() - started) * 1000.0
        return {
            "pipeline": "telemetry -> features -> anomaly+prediction -> risk -> rca -> remediation -> recovery",
            "rows": int(len(raw)),
            "services": sorted(raw["service"].unique().tolist()) if "service" in raw.columns else [],
            "n_features": int(len(engine.feature_columns)),
            "threshold": self.threshold,
            "mode": resolved_mode,
            "detection": {
                "n_flagged": flagged,
                "n_services": int(latest.shape[0]),
                "latest": _latest_rows(latest),
            },
            "remediation": remediation,
            "recovery": recovery_out,
            "timings_ms": {k: round(v, 2) for k, v in timings.items()},
        }


def build_services(names: list[str], services_path: str | Path | None = None) -> list[ServiceSimulator]:
    known: dict[str, dict] = {}
    if services_path is not None:
        config = load_settings(services_path)
        known = {item["name"]: item for item in config.get("services", [])}
    services = []
    for name in names:
        if name in known:
            item = known[name]
            services.append(
                ServiceSimulator(name=name, replicas=item.get("replicas", 1), profile=item.get("profile", {}))
            )
        else:
            services.append(ServiceSimulator(name=name))
    return services


def generate_incident(
    names: list[str],
    scenario_type: str | None,
    duration_min: float = 45,
    recovery_min: float = 20,
    interval_sec: int = 30,
    seed: int = 42,
    services_path: str | Path | None = None,
) -> dict:
    services = build_services(names, services_path)
    start = datetime.now(timezone.utc).replace(microsecond=0)
    scenarios: list[ScenarioSpec] = []
    if scenario_type:
        scenarios = [
            ScenarioSpec(service=services[0].name, scenario_type=scenario_type, start_offset_min=0.0, duration_min=duration_min)
        ]

    before_rows: list[dict] = []
    incidents: list[dict] = []
    generator = TelemetryGenerator(interval_sec=interval_sec, seed=seed, start_ts=start)
    for batch in generator.iter_steps(services, scenarios, duration_min=duration_min):
        for metric in batch.metric_rows:
            before_rows.append(metric.to_dict())
        for record in batch.incident_ends:
            incidents.append(record.to_dict())

    after_rows: list[dict] = []
    second_start = start + timedelta(minutes=duration_min)
    generator2 = TelemetryGenerator(interval_sec=interval_sec, seed=seed + 101, start_ts=second_start)
    for batch in generator2.iter_steps(services, [], duration_min=recovery_min):
        for metric in batch.metric_rows:
            after_rows.append(metric.to_dict())

    return {
        "service": services[0].name,
        "scenario_type": scenario_type,
        "interval_sec": interval_sec,
        "start_ts": start.isoformat(),
        "before_rows": before_rows,
        "after_rows": after_rows,
        "incidents": incidents,
    }


def configured_scenarios(settings_path: str | Path) -> dict:
    specs = load_scenarios(settings_path)
    services = load_services(settings_path)
    names = [service.name for service in services]
    return {
        "services": names,
        "scenarios": [
            {
                "service": spec.service,
                "scenario_type": spec.scenario_type,
                "start_offset_min": spec.start_offset_min,
                "duration_min": spec.duration_min,
                "intensity": spec.intensity,
            }
            for spec in specs
        ],
    }