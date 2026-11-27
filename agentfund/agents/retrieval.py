"""Point-in-time document retrieval (filings, transcripts).

Documents carry a ``filed_at`` date and are only retrievable from views on or
after that date. Ranking is a small TF-IDF over paragraph chunks: enough to pick
the relevant passages of a 10-K without an embedding service. The EDGAR loader
that fills this store comes in a later slice.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import pandas as pd

_WORD = re.compile(r"[a-z][a-z0-9\-]+")


@dataclass(frozen=True)
class Document:
    doc_id: str
    ticker: str
    filed_at: pd.Timestamp
    kind: str  # e.g. "10-K", "10-Q", "earnings_call"
    text: str


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    kind: str
    filed_at: pd.Timestamp
    text: str


def _tokens(s: str) -> list[str]:
    return _WORD.findall(s.lower())


def _chunks(doc: Document, size: int = 900) -> list[Chunk]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", doc.text) if p.strip()]
    out, buf = [], ""
    for p in paras:
        if buf and len(buf) + len(p) > size:
            out.append(Chunk(doc.doc_id, doc.kind, doc.filed_at, buf))
            buf = ""
        buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        out.append(Chunk(doc.doc_id, doc.kind, doc.filed_at, buf))
    return out


class DocumentStore:
    def __init__(self, docs: list[Document] | None = None):
        self._chunks: list[Chunk] = []
        self._ticker: list[str] = []
        for d in docs or []:
            self.add(d)

    def add(self, doc: Document) -> None:
        doc = Document(doc.doc_id, doc.ticker, pd.Timestamp(doc.filed_at).normalize(), doc.kind, doc.text)
        for c in _chunks(doc):
            self._chunks.append(c)
            self._ticker.append(doc.ticker)

    def search(self, ticker: str, as_of, query: str, k: int = 6) -> list[Chunk]:
        as_of = pd.Timestamp(as_of)
        pool = [c for c, t in zip(self._chunks, self._ticker) if t == ticker and c.filed_at <= as_of]
        if not pool:
            return []
        docs_tokens = [Counter(_tokens(c.text)) for c in pool]
        df = Counter(w for toks in docs_tokens for w in toks)
        n = len(pool)
        q = set(_tokens(query))

        def score(i: int) -> float:
            toks = docs_tokens[i]
            length = sum(toks.values()) or 1
            s = sum((toks[w] / length) * math.log(1 + n / df[w]) for w in q if w in toks)
            return s + 1e-6 * pool[i].filed_at.value / 1e18  # tie-break toward recent filings

        ranked = sorted(range(n), key=score, reverse=True)[:k]
        return [pool[i] for i in ranked]
