from __future__ import annotations

import re
from typing import Any

from .schema import CAUSE_CATEGORIES, RCAResult

_MESSAGE_KEYWORDS = {
    "memory_leak": [
        "memory", "gc", "heap", "oom", "out of memory", "outofmemory", "leak",
        "allocation", "metaspace",
    ],
    "connection_pool_exhaustion": [
        "connection pool", "pool exhausted", "exhaustion", "connections", "jdbc",
        "acquire", "max pool", "borrow", "timeout acquiring",
    ],
    "latency_spike": [
        "latency", "p99", "p95", "queue depth", "queue buildup", "slow query",
        "request timeout", "saturated", "blocked thread",
    ],
    "network_partition": [
        "network", "unreachable", "refused", "partition", "reconnect", "downstream",
        "connect reset", "no route", "host is down",
    ],
    "disk_fill": [
        "disk", "space", "no space left", "device", "filesystem", "storage",
        "inode", "write error",
    ],
    "code_deployment": ["deployment", "release", "rollout", "v2.", "canary", "rollback", "config change"],
    "traffic_spike": ["traffic", "load spike", "qps", "requests per second", "autoscaler", "scaled up"],
}

_FEATURE_TO_CATEGORY = {
    "memory_leak": {"memory_growth", "cpu_memory", "memory"},
    "connection_pool_exhaustion": {"request_rate", "db_usage", "error_rate", "latency_per_request"},
    "latency_spike": {"latency", "latency_per_request"},
    "network_partition": {"network_in", "network_out"},
    "disk_fill": {"disk", "disk_growth"},
}


def _evidence_text(evidence: dict[str, Any] | str) -> str:
    import json

    if isinstance(evidence, str):
        return evidence.lower()
    if isinstance(evidence, dict):
        blob = json.dumps(evidence, default=str)
    else:
        blob = str(evidence)
    return blob.lower()


def _feature_matches(name: str, category: str) -> bool:
    lowered = name.lower().replace("-", "_")
    for token in _FEATURE_TO_CATEGORY.get(category, ()):
        if token in lowered:
            return True
    return False


def _score_memory_component(evidence: dict[str, Any]) -> float:
    components = evidence.get("components", {}) or {}
    return float(components.get("memory_growth", 0.0))


def _evidence_list(evidence: dict[str, Any] | str) -> list[dict]:
    if isinstance(evidence, str):
        return []
    if isinstance(evidence, dict) and isinstance(evidence.get("correlated_logs"), list):
        return evidence.get("correlated_logs", [])
    return []


def classify(evidence: dict[str, Any] | str) -> RCAResult:
    """Deterministic fallback RCA: score each cause category and pick the argmax.

    Uses (a) keyword hits in correlated log messages, (b) anomalous feature names,
    and (c) the memory_growth component for memory-leak discrimination.
    """
    text = _evidence_text(evidence)
    component = _score_memory_component(evidence) if isinstance(evidence, dict) else 0.0

    scores: dict[str, float] = {}
    for category in CAUSE_CATEGORIES:
        if category in ("unknown",):
            continue
        score = 0.0

        for keyword in _MESSAGE_KEYWORDS.get(category, ()):
            if keyword in text:
                score += 2.0

        features = []
        if isinstance(evidence, dict):
            features = evidence.get("top_anomalous_features", []) or []
            if not isinstance(features, list):
                features = []
        for feat in features:
            name = feat.get("feature", "") if isinstance(feat, dict) else str(feat)
            if name and _feature_matches(str(name), category):
                score += 1.5

        if category == "memory_leak" and component > 0.75:
            score += 1.0

        scores[category] = score

    best = max(scores, key=scores.get)
    best_score = scores[best]
    total = sum(scores.values()) or 1.0
    confidence = round(min(0.98, 0.3 + best_score / total), 2)

    if best_score <= 0:
        best = "unknown"
        confidence = 0.3

    return RCAResult(
        probable_root_cause=_summary(best),
        confidence_score=confidence,
        recommended_action=_action(best),
        root_cause_category=best,
        evidence=_evidence_snippets(evidence),
    )


def _summary(category: str) -> str:
    summaries = {
        "memory_leak": "Sustained memory growth and GC pressure consistent with a memory leak in the service runtime.",
        "connection_pool_exhaustion": "Connection pool exhaustion: demand exceeded available downstream connections, causing acquire timeouts.",
        "latency_spike": "Latency spike driven by an exhausted or saturated upstream path (queue buildup / slow processing).",
        "network_partition": "Network partition or downstream unreachability: connectivity loss between the service and its dependencies.",
        "disk_fill": "Disk capacity filling: filesystem pressure from log or storage growth on the node.",
        "code_deployment": "Behavior change correlated with a recent deployment, indicating a regression.",
        "traffic_spike": "Unusual traffic surge overwhelming current capacity.",
        "unknown": "No single dominant signal; further investigation required.",
    }
    return summaries.get(category, "Undetermined root cause.")


def _action(category: str) -> str:
    actions = {
        "memory_leak": "Trigger a heap dump and memory-profile analysis; roll the service if confirmed.",
        "connection_pool_exhaustion": "Increase or tune the connection pool limit and check downstream latency; restart pools.",
        "latency_spike": "Scale horizontally to reduce queue depth and enable request shedding on non-critical paths.",
        "network_partition": "Verify service mesh / firewall / DNS; check for flapping replicas and downstream health.",
        "disk_fill": "Free disk space, rotate logs, and add a disk-utilisation alarm for earlier warning.",
        "code_deployment": "Roll back the most recent deployment to mitigate while the regression is analysed.",
        "traffic_spike": "Trigger autoscaling and engage rate limiting to absorb the surge.",
        "unknown": "Collect additional traces and correlate upstream dependency health.",
    }
    return actions.get(category, "Review dashboards and traces before acting.")


def _evidence_snippets(evidence: dict[str, Any] | str) -> list[str]:
    snippets: list[str] = []

    if isinstance(evidence, str):
        snippets.append("evidence package received as JSON string")
        return snippets

    components = evidence.get("components", {}) or {}
    for name, value in components.items():
        if isinstance(value, (int, float)):
            snippets.append(f"component {name} = {value:.3f}")

    deltas = evidence.get("metrics_deltas", {}) or {}
    for name, value in list(deltas.items())[:4]:
        if value is not None:
            snippets.append(f"metric {name} delta = {value}")

    features = evidence.get("top_anomalous_features", []) or []
    for feat in features[:4]:
        if isinstance(feat, dict):
            snippets.append(
                f"feature {feat.get('feature')} z={feat.get('zscore')} ({feat.get('direction')})"
            )

    for log in _evidence_list(evidence)[:4]:
        msg = log.get("message") if isinstance(log, dict) else str(log)
        snippets.append(f"log: {msg}")

    return snippets[:8] if snippets else ["no structured evidence available"]