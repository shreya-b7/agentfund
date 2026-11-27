"""Deterministic risk officer: approves, resizes or vetoes proposed target weights.

The officer applies rules in a fixed order and records every breach, so each
decision can be reproduced exactly from its inputs. An LLM risk agent can sit in
front of this (to argue for vetoes or tighter limits), but the final weights
always pass through :meth:`RiskOfficer.review`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .limits import RiskLimits


class Verdict(str, Enum):
    APPROVED = "approved"
    RESIZED = "resized"
    VETOED = "vetoed"


@dataclass(frozen=True)
class Breach:
    rule: str
    ticker: str | None
    requested: float
    allowed: float


@dataclass
class RiskContext:
    current_weights: dict[str, float]
    prices: dict[str, float]
    adv_dollars: dict[str, float]  # 20-day average dollar volume
    sectors: dict[str, str]
    tradeable: set[str]
    equity: float
    drawdown: float  # current drawdown from peak, as a positive fraction


@dataclass
class RiskDecision:
    verdict: Verdict
    proposed: dict[str, float]
    approved: dict[str, float]
    breaches: list[Breach] = field(default_factory=list)
    limits_fingerprint: str = ""

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "proposed": {k: round(v, 6) for k, v in sorted(self.proposed.items())},
            "approved": {k: round(v, 6) for k, v in sorted(self.approved.items())},
            "breaches": [b.__dict__ for b in self.breaches],
            "limits_fingerprint": self.limits_fingerprint,
        }


_EPS = 1e-9


class RiskOfficer:
    def __init__(self, limits: RiskLimits | None = None):
        self._limits = limits or RiskLimits()

    @property
    def limits(self) -> RiskLimits:
        return self._limits

    def review(self, proposed: dict[str, float], ctx: RiskContext) -> RiskDecision:
        L = self._limits
        breaches: list[Breach] = []
        w = {k: float(v) for k, v in proposed.items() if abs(v) > _EPS}
        cur = {k: float(v) for k, v in ctx.current_weights.items() if abs(v) > _EPS}

        # 0. Kill switch: drawdown beyond the limit -> flatten everything.
        if ctx.drawdown > L.drawdown_kill_switch:
            breaches.append(Breach("drawdown_kill_switch", None, ctx.drawdown, L.drawdown_kill_switch))
            return RiskDecision(Verdict.VETOED, dict(proposed), {}, breaches, L.fingerprint())

        # 1. Untradeable names (delisted / not in universe) can only be exited.
        for t in list(w):
            if t not in ctx.tradeable:
                breaches.append(Breach("not_tradeable", t, w[t], 0.0))
                del w[t]

        # 2. Long-only.
        if L.long_only:
            for t in list(w):
                if w[t] < 0:
                    breaches.append(Breach("long_only", t, w[t], 0.0))
                    del w[t]

        # 3. Minimum price for new or increased positions.
        for t in list(w):
            px = ctx.prices.get(t, 0.0)
            if px < L.min_price and abs(w[t]) > abs(cur.get(t, 0.0)) + _EPS:
                allowed = cur.get(t, 0.0)
                breaches.append(Breach("min_price", t, w[t], allowed))
                w[t] = allowed

        # 4. Per-position cap.
        for t, v in w.items():
            if abs(v) > L.max_position_weight + _EPS:
                capped = L.max_position_weight * (1 if v > 0 else -1)
                breaches.append(Breach("max_position_weight", t, v, capped))
                w[t] = capped

        # 5. Sector cap: scale names in an over-weight sector pro rata.
        by_sector: dict[str, float] = {}
        for t, v in w.items():
            by_sector[ctx.sectors.get(t, "?")] = by_sector.get(ctx.sectors.get(t, "?"), 0.0) + abs(v)
        for s, gross in by_sector.items():
            if gross > L.max_sector_weight + _EPS:
                k = L.max_sector_weight / gross
                breaches.append(Breach("max_sector_weight", s, gross, L.max_sector_weight))
                for t in w:
                    if ctx.sectors.get(t, "?") == s:
                        w[t] *= k

        # 6. Gross exposure cap.
        gross = sum(abs(v) for v in w.values())
        if gross > L.max_gross_exposure + _EPS:
            k = L.max_gross_exposure / gross
            breaches.append(Breach("max_gross_exposure", None, gross, L.max_gross_exposure))
            w = {t: v * k for t, v in w.items()}

        # 7. Liquidity: cap each trade at a fraction of average dollar volume.
        for t in sorted(set(w) | set(cur)):
            if t not in ctx.tradeable:
                continue  # forced exits are always allowed
            dw = w.get(t, 0.0) - cur.get(t, 0.0)
            max_dw = L.max_adv_participation * ctx.adv_dollars.get(t, 0.0) / max(ctx.equity, _EPS)
            if abs(dw) > max_dw + _EPS:
                new = cur.get(t, 0.0) + max_dw * (1 if dw > 0 else -1)
                breaches.append(Breach("max_adv_participation", t, w.get(t, 0.0), new))
                w[t] = new

        # 8. Turnover: blend toward current weights if the rebalance is too large.
        names = set(w) | set(cur)
        turnover = sum(abs(w.get(t, 0.0) - cur.get(t, 0.0)) for t in names) / 2
        if turnover > L.max_turnover + _EPS:
            k = L.max_turnover / turnover
            breaches.append(Breach("max_turnover", None, turnover, L.max_turnover))
            w = {t: cur.get(t, 0.0) + k * (w.get(t, 0.0) - cur.get(t, 0.0)) for t in names}
            # Forced exits of untradeable names are not blended.
            for t in names:
                if t not in ctx.tradeable:
                    w[t] = 0.0

        w = {t: v for t, v in w.items() if abs(v) > _EPS}
        if not breaches:
            verdict = Verdict.APPROVED
        elif not w and proposed:
            verdict = Verdict.VETOED
        else:
            verdict = Verdict.RESIZED
        return RiskDecision(verdict, dict(proposed), w, breaches, L.fingerprint())
