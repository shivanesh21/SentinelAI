from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

CAUSE_CATEGORIES = (
    "memory_leak",
    "connection_pool_exhaustion",
    "latency_spike",
    "network_partition",
    "disk_fill",
    "code_deployment",
    "traffic_spike",
    "unknown",
)


@dataclass
class RCAResult:
    probable_root_cause: str
    confidence_score: float
    recommended_action: str = ""
    root_cause_category: str = "unknown"
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, data: dict) -> "RCAResult":
        required = ("probable_root_cause", "confidence_score", "recommended_action")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"missing required schema keys: {missing}")
        cause = str(data["probable_root_cause"]).strip()
        if not cause:
            raise ValueError("probable_root_cause must be a non-empty string")
        confidence = float(data["confidence_score"])
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence_score must be in [0, 1]")
        category = str(data.get("root_cause_category", "unknown"))
        if category not in CAUSE_CATEGORIES:
            # guardrail: unknown/typo'd categories from the LLM are coerced,
            # never propagated up the stack
            category = "unknown"
        return cls(
            probable_root_cause=cause,
            confidence_score=confidence,
            recommended_action=str(data["recommended_action"]).strip(),
            root_cause_category=category,
            evidence=[str(item) for item in data.get("evidence", [])],
        )

    @classmethod
    def from_json(cls, text: str) -> "RCAResult":
        return cls.from_dict(json.loads(text))