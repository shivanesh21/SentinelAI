from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any

import requests

from .audit import AuditLogger, RCAEvent, new_request_id
from .heuristic import classify as heuristic_classify
from .prompt import build_rca_prompt
from .schema import RCAResult

DEFAULT_ANTHROPIC_MODEL = "claude-3-5-sonnet-latest"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_LOCAL_MODEL = "llama3"


def resolve_provider() -> str:
    """Resolve the active provider from the environment (matches .env.example).

    Returns one of: anthropic | openai | local | heuristic.
    A provider is chosen only when it is declared AND a usable credential/endpoint
    is present; otherwise we fall back to the deterministic heuristic, so RCA
    always returns a result without external dependencies.
    """
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if provider == "openai" and os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if provider == "local":
        return "local"
    return "heuristic"


def extract_json(text: str) -> dict:
    """Best-effort JSON extraction from an LLM response (handles stray ``` fences)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first == -1 or last == -1 or last < first:
        raise ValueError("no JSON object found in LLM response")
    return json.loads(cleaned[first : last + 1])


def parse_result(raw: str) -> RCAResult:
    """Parse and schema-validate an LLM response; raises on malformed output."""
    return RCAResult.from_dict(extract_json(raw))


class RCAClient:
    """Call the configured LLM to analyse an evidence package, falling back to the
    deterministic heuristic when no provider is configured, the call fails, or the
    output fails schema validation. Every call is audited (input evidence + output
    diagnosis) when an AuditLogger is attached."""

    def __init__(self, provider: str | None = None, audit: AuditLogger | None = None):
        self.provider = provider or resolve_provider()
        self.audit = audit

    def analyze(
        self,
        evidence: dict[str, Any] | str,
        request_id: str | None = None,
        audit: AuditLogger | None = None,
    ) -> RCAResult:
        logger = audit or self.audit
        start = time.perf_counter()

        event = RCAEvent(
            request_id=request_id or new_request_id(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            provider=self.provider,
            service=evidence.get("service", "?") if isinstance(evidence, dict) else "?",
            evidence=evidence if isinstance(evidence, dict) else {"evidence": str(evidence)},
            status="ok",
        )

        if self.provider == "heuristic":
            result = heuristic_classify(evidence)
            self._finish(event, result, start, logger)
            return result

        try:
            if self.provider == "anthropic":
                raw = self._call_anthropic(evidence)
            elif self.provider == "openai":
                raw = self._call_openai(evidence)
            elif self.provider == "local":
                raw = self._call_local(evidence)
            else:
                result = heuristic_classify(evidence)
                self._finish(event, result, start, logger)
                return result
            event.llm_raw = raw
            result = parse_result(raw)
            self._finish(event, result, start, logger)
            return result
        except Exception as exc:
            event.status = "fallback_heuristic"
            event.error = str(exc)
            result = heuristic_classify(evidence)
            self._finish(event, result, start, logger)
            print(f"[rca] LLM call/output failed ({self.provider}): {exc}; using heuristic fallback")
            return result

    @staticmethod
    def _finish(event: RCAEvent, result: RCAResult, start: float, logger: AuditLogger | None) -> None:
        event.duration_ms = round((time.perf_counter() - start) * 1000, 3)
        event.result = result.to_dict()
        if logger is not None:
            logger.record(event)

    def _call_anthropic(self, evidence: dict[str, Any]) -> str:
        import anthropic

        system, user = build_rca_prompt(evidence)
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = client.messages.create(
            model=os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL),
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text")

    def _call_openai(self, evidence: dict[str, Any]) -> str:
        from openai import OpenAI

        system, user = build_rca_prompt(evidence)
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content

    def _call_local(self, evidence: dict[str, Any]) -> str:
        system, user = build_rca_prompt(evidence)
        url = os.environ.get("LOCAL_LLM_URL", "http://localhost:11434").rstrip("/")
        response = requests.post(
            f"{url}/api/generate",
            json={
                "model": os.environ.get("LLM_MODEL", DEFAULT_LOCAL_MODEL),
                "prompt": f"{system}\n\n{user}",
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"]