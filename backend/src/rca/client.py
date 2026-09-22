from __future__ import annotations

import json
import os
from typing import Any

import requests

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


class RCAClient:
    """Call the configured LLM to analyse an evidence package, falling back to the
    deterministic heuristic when no provider is configured or the call fails."""

    def __init__(self, provider: str | None = None):
        self.provider = provider or resolve_provider()

    def analyze(self, evidence: dict[str, Any] | str) -> RCAResult:
        system, user = build_rca_prompt(evidence)
        if self.provider == "heuristic":
            return heuristic_classify(evidence)
        try:
            if self.provider == "anthropic":
                raw = self._call_anthropic(system, user)
            elif self.provider == "openai":
                raw = self._call_openai(system, user)
            elif self.provider == "local":
                raw = self._call_local(system, user)
            else:
                return heuristic_classify(evidence)
            return RCAResult.from_json(raw)
        except Exception as exc:  # pragma: no cover - network/api failures
            print(f"[rca] LLM call failed ({self.provider}): {exc}; using heuristic fallback")
            return heuristic_classify(evidence)

    def _call_anthropic(self, system: str, user: str) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = client.messages.create(
            model=os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL),
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text")

    def _call_openai(self, system: str, user: str) -> str:
        from openai import OpenAI

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

    def _call_local(self, system: str, user: str) -> str:
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


def extract_json(text: str) -> dict:
    """Best-effort JSON extraction from an LLM response (handles stray ``` fences)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        first = cleaned.find("{")
        last = cleaned.rfind("}")
        cleaned = cleaned[first : last + 1]
    return json.loads(cleaned)