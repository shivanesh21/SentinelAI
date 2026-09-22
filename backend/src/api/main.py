from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.features.store import FeatureStore
from src.risk import RiskEngine
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


def _get_risk_engine() -> RiskEngine:
    global _risk_engine
    if _risk_engine is None:
        _risk_engine = RiskEngine.load(BACKEND / "models", settings)
    return _risk_engine


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


if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")