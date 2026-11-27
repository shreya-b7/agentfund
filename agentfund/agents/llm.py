"""LLM clients with per-call cost and latency tracking.

``AnthropicLLM`` calls Claude with structured outputs. ``OfflineLLM`` is a
deterministic stand-in used in tests and demos when no API key is set; it builds
a valid response from the prompt's numbers with a fixed rule, so the pipeline,
logging and evals can all run without network access.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "claude-opus-5-5"
# USD per million tokens (input, output). Update when prices change.
PRICES = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "offline-stub": (0.0, 0.0),
}


@dataclass
class CallRecord:
    agent: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float
    stop_reason: str
    at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CallLedger:
    """Collects one :class:`CallRecord` per LLM call; optionally appends them to a JSONL file."""

    def __init__(self, path: str | Path | None = None):
        self.records: list[CallRecord] = []
        self.path = Path(path) if path else None

    def add(self, rec: CallRecord) -> None:
        self.records.append(rec)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(json.dumps(asdict(rec)) + "\n")

    def totals(self) -> dict[str, float]:
        n = len(self.records)
        return {
            "calls": n,
            "cost_usd": sum(r.cost_usd for r in self.records),
            "input_tokens": sum(r.input_tokens for r in self.records),
            "output_tokens": sum(r.output_tokens for r in self.records),
            "mean_latency_s": (sum(r.latency_s for r in self.records) / n) if n else 0.0,
        }


class LLMRefusal(RuntimeError):
    pass


class LLM(Protocol):
    model: str

    def structured(self, *, agent: str, system: str, user: str, schema: type[T]) -> T: ...


def _cost(model: str, inp: int, out: int) -> float:
    pin, pout = PRICES.get(model, (0.0, 0.0))
    return (inp * pin + out * pout) / 1e6


class AnthropicLLM:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = "medium",
        max_tokens: int = 16000,
        ledger: CallLedger | None = None,
    ):
        import anthropic  # imported lazily so offline use needs no key

        self.client = anthropic.Anthropic()
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.ledger = ledger or CallLedger()

    def structured(self, *, agent: str, system: str, user: str, schema: type[T]) -> T:
        t0 = time.perf_counter()
        resp = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            output_format=schema,
            # If a safety classifier declines, the API retries on a fallback model in the same call.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        latency = time.perf_counter() - t0
        u = resp.usage
        self.ledger.add(
            CallRecord(
                agent=agent,
                model=resp.model,
                input_tokens=u.input_tokens,
                output_tokens=u.output_tokens,
                latency_s=latency,
                cost_usd=_cost(self.model, u.input_tokens, u.output_tokens),
                stop_reason=str(resp.stop_reason),
            )
        )
        if resp.stop_reason == "refusal":
            raise LLMRefusal(f"{agent}: model declined ({getattr(resp, 'stop_details', None)})")
        if resp.stop_reason == "max_tokens" or resp.parsed_output is None:
            raise RuntimeError(f"{agent}: no structured output (stop_reason={resp.stop_reason})")
        return resp.parsed_output


class OfflineLLM:
    """Deterministic stub. ``responder(agent, user_prompt, schema)`` returns the parsed object."""

    model = "offline-stub"

    def __init__(self, responder: Callable[[str, str, type], BaseModel], ledger: CallLedger | None = None):
        self.responder = responder
        self.ledger = ledger or CallLedger()

    def structured(self, *, agent: str, system: str, user: str, schema: type[T]) -> T:
        t0 = time.perf_counter()
        out = self.responder(agent, user, schema)
        approx_in = (len(system) + len(user)) // 4
        approx_out = len(out.model_dump_json()) // 4
        self.ledger.add(
            CallRecord(agent, self.model, approx_in, approx_out, time.perf_counter() - t0, 0.0, "end_turn")
        )
        return schema.model_validate(out.model_dump())


def default_llm(ledger: CallLedger | None = None, offline: bool | None = None) -> LLM:
    """Claude if credentials are configured (and ``offline`` is not forced), else the stub."""
    from .analyst import offline_analyst_responder

    if offline is None:
        offline = not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    if offline:
        return OfflineLLM(offline_analyst_responder, ledger)
    return AnthropicLLM(ledger=ledger)
