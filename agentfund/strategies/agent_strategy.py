"""Strategy that turns analyst research notes into target weights.

Monthly, the analyst covers the ``coverage`` most liquid names; names rated buy
or better are weighted by their signed score. The risk officer still has the
final say on every rebalance (the engine enforces that).
"""

from __future__ import annotations

from ..agents.analyst import AnalystAgent
from ..data.pit import PITView
from .baselines import _Monthly


class AnalystStrategy(_Monthly):
    def __init__(self, analyst: AnalystAgent, coverage: int = 15, min_score: float = 0.2):
        super().__init__()
        self.analyst, self.coverage, self.min_score = analyst, coverage, min_score
        self.name = "analyst_anon" if analyst.anonymize else "analyst_named"
        self.notes: list = []

    def weights(self, view: PITView) -> dict[str, float]:
        dv = (view.history("close", 20) * view.history("volume", 20)).mean()
        hist = view.history("close", 253).notna().sum()
        eligible = [t for t in view.tickers() if hist.get(t, 0) >= 253]
        names = dv.reindex(eligible).dropna().sort_values(ascending=False).index[: self.coverage]
        scores = {}
        for t in names:
            note = self.analyst.research(view, t)
            self.notes.append(note)
            if note.score >= self.min_score:
                scores[t] = note.score
        total = sum(scores.values())
        return {t: s / total for t, s in scores.items()} if total > 0 else {}
