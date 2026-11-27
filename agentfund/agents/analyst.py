"""Analyst agent: turns point-in-time data and filings into a structured research note.

The agent only sees a "data pack" built from a :class:`PITView`, so it cannot be
*handed* future data. It can still *remember* the future from training, which is
the look-ahead contamination this project measures. ``anonymize=True`` removes
the cues that let a model recall what happened next: the ticker and company
name become an alias, prices are rebased to 100 and calendar dates become
offsets from the decision date. Comparing named and anonymized runs on the same
dates estimates how much of an agent's backtest edge comes from memory.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.pit import PITView
from .llm import LLM
from .retrieval import DocumentStore
from .schemas import Evidence, ResearchNote

SYSTEM_PROMPT = """You are an equity research analyst at a long-only fund. You write one research note per request.

Use only the materials in the request. The decision date is the last day of information; treat everything after it as unknown, even if you believe you remember what happened. If you notice that your view depends on anything outside the materials, set used_outside_knowledge to true.

Base the rating on the evidence: price trend and volatility, the direction of revenue and earnings in the most recent filings, and anything material in the filing excerpts. Hold is the right answer when the evidence is mixed. Conviction should reflect how strong and consistent the evidence is, not how interesting the story is. Each evidence item must cite a fact that appears in the materials."""

QUERY = "revenue growth guidance margin outlook risk demand competition liquidity impairment"


def alias_for(ticker: str, salt: str = "agentfund") -> str:
    return "CO-" + hashlib.sha256(f"{salt}:{ticker}".encode()).hexdigest()[:6].upper()


def build_data_pack(
    view: PITView,
    ticker: str,
    docs: DocumentStore | None = None,
    anonymize: bool = False,
    quarters: int = 6,
) -> dict:
    px = view.history("close", lookback=253)[ticker].dropna()
    if len(px) < 22:
        raise ValueError(f"not enough price history for {ticker} as of {view.as_of.date()}")
    r = px.pct_change().dropna()

    def ret(n: int) -> float | None:
        return round(float(px.iloc[-1] / px.iloc[-1 - n] - 1), 4) if len(px) > n else None

    uni = view.universe()
    row = uni[uni["ticker"] == ticker]
    name = row["name"].iloc[0] if len(row) else ticker
    sector = row["sector"].iloc[0] if len(row) else "unknown"

    def when(d: pd.Timestamp) -> str:
        return f"T-{(view.as_of - d).days}d" if anonymize else str(d.date())

    fund = view.fundamentals(ticker).tail(quarters + 4)
    rows = []
    for i, f in enumerate(fund.itertuples()):
        prev_year = fund.iloc[i - 4] if i >= 4 else None
        yoy = None
        if prev_year is not None and getattr(f, "revenue", None) and prev_year["revenue"]:
            yoy = round(float(f.revenue / prev_year["revenue"] - 1), 4)
        rows.append(
            {
                "period": f"Q-{len(fund) - i}" if anonymize else str(f.period_end.date()),
                "filed": when(f.filed_date),
                **{
                    c: round(float(getattr(f, c)), 4)
                    for c in fund.columns
                    if c not in ("ticker", "period_end", "filed_date") and pd.notna(getattr(f, c))
                },
                "revenue_yoy": yoy,
            }
        )
    rows = rows[-quarters:]

    excerpts = []
    if docs is not None:
        for c in docs.search(ticker, view.as_of, QUERY):
            text = c.text
            if anonymize:
                text = re.sub(re.escape(ticker), "the company", text, flags=re.I)
                text = re.sub(re.escape(name), "the company", text, flags=re.I)
                text = re.sub(r"\b(19|20)\d{2}\b", "[year]", text)
            excerpts.append({"source": f"{c.kind} filed {when(c.filed_at)}", "text": text})

    level = 100 * px.iloc[-1] / px.iloc[0] if anonymize else float(px.iloc[-1])
    return {
        "company": alias_for(ticker) if anonymize else f"{name} ({ticker})",
        "sector": sector,
        "decision_date": "T" if anonymize else str(view.as_of.date()),
        "price": {
            "last": round(float(level), 2),
            "rebased_to_100_one_year_ago": anonymize,
            "return_1m": ret(21),
            "return_3m": ret(63),
            "return_6m": ret(126),
            "return_12m": ret(252),
            "volatility_annualized": round(float(r.tail(63).std() * np.sqrt(252)), 4),
            "drawdown_from_52w_high": round(float(1 - px.iloc[-1] / px.max()), 4),
        },
        "fundamentals": rows,
        "filing_excerpts": excerpts,
    }


def render_prompt(pack: dict) -> str:
    return (
        f"Write a research note on {pack['company']} as of decision date {pack['decision_date']}.\n\n"
        f"<materials>\n{json.dumps(pack, indent=2)}\n</materials>"
    )


class AnalystAgent:
    name = "analyst"

    def __init__(self, llm: LLM, docs: DocumentStore | None = None, anonymize: bool = False, cache_dir: str | Path | None = None):
        self.llm, self.docs, self.anonymize = llm, docs, anonymize
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def _cache_path(self, prompt: str) -> Path | None:
        if not self.cache_dir:
            return None
        key = hashlib.sha256(f"{self.llm.model}|{SYSTEM_PROMPT}|{prompt}".encode()).hexdigest()[:24]
        return self.cache_dir / f"{key}.json"

    def research(self, view: PITView, ticker: str) -> ResearchNote:
        pack = build_data_pack(view, ticker, self.docs, self.anonymize)
        prompt = render_prompt(pack)
        cp = self._cache_path(prompt)
        if cp and cp.exists():
            return ResearchNote.model_validate_json(cp.read_text())
        note = self.llm.structured(agent=self.name, system=SYSTEM_PROMPT, user=prompt, schema=ResearchNote)
        # Identity and date come from our side, never from the model.
        note = note.model_copy(update={"ticker": ticker, "as_of": str(view.as_of.date())})
        if cp:
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(note.model_dump_json(indent=2))
        return note


def offline_analyst_responder(agent: str, user: str, schema: type) -> ResearchNote:
    """Deterministic rule standing in for the LLM: trend plus revenue growth."""
    pack = json.loads(user.split("<materials>\n", 1)[1].rsplit("\n</materials>", 1)[0])
    p = pack["price"]
    mom = (p.get("return_12m") or 0.0) - (p.get("return_1m") or 0.0)
    yoys = [q["revenue_yoy"] for q in pack["fundamentals"] if q.get("revenue_yoy") is not None]
    growth = yoys[-1] if yoys else 0.0
    score = 0.6 * np.tanh(mom / 0.3) + 0.4 * np.tanh(growth / 0.15)
    rating = (
        "strong_buy" if score > 0.5 else "buy" if score > 0.15 else "hold" if score > -0.15 else "sell" if score > -0.5 else "strong_sell"
    )
    evidence = [Evidence(source="price", fact=f"12-month return {p.get('return_12m')}")]
    if yoys:
        evidence.append(Evidence(source=f"fundamentals {pack['fundamentals'][-1]['period']}", fact=f"revenue YoY {growth}"))
    return ResearchNote(
        ticker="?",
        as_of=pack["decision_date"],
        rating=rating,
        conviction=round(float(min(1.0, abs(score) + 0.2)), 3),
        expected_return_3m=round(float(0.05 * score), 4),
        thesis=f"[offline stub] Trend score {mom:+.2f} and revenue growth {growth:+.2%} give a combined score of {score:+.2f}.",
        key_drivers=["12-1 month price trend", "latest revenue growth"],
        risks=[f"volatility {p.get('volatility_annualized')}", f"drawdown from high {p.get('drawdown_from_52w_high')}"],
        evidence=evidence,
        used_outside_knowledge=False,
    )
