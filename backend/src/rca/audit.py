from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_AUDIT_PATH = "data/rca_audit.jsonl"


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class RCAEvent:
    """One audited RCA call: the input evidence and the output diagnosis."""

    request_id: str
    timestamp: str
    provider: str
    service: str
    evidence: dict[str, Any]
    result: dict[str, Any] | None = None
    status: str = "ok"                 # ok | fallback_heuristic | error
    duration_ms: float = 0.0
    error: str | None = None
    llm_raw: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["error"] is None:
            data.pop("error")
        if data["llm_raw"] is None:
            data.pop("llm_raw")
        return data


class AuditLogger:
    """Append-only JSONL audit trail recording every RCA call."""

    def __init__(self, path: str | Path = DEFAULT_AUDIT_PATH):
        self.path = Path(path)

    def record(self, event: RCAEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event.to_dict(), default=str)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def read(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return up to `limit` most recent audit records (oldest first)."""
        if limit <= 0 or not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records[-limit:]