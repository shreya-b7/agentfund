"""Structured outputs produced by agents."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Rating = Literal["strong_sell", "sell", "hold", "buy", "strong_buy"]
RATING_SCORE: dict[str, float] = {"strong_sell": -1.0, "sell": -0.5, "hold": 0.0, "buy": 0.5, "strong_buy": 1.0}


class Evidence(BaseModel):
    source: str = Field(description="Which provided material this comes from, e.g. 'fundamentals 2019Q1' or a document id")
    fact: str = Field(description="The specific fact or figure, stated as in the source")


class ResearchNote(BaseModel):
    ticker: str
    as_of: str = Field(description="Decision date (YYYY-MM-DD). Nothing after this date may be used.")
    rating: Rating
    conviction: float = Field(description="0 to 1: how confident the analyst is in the rating")
    expected_return_3m: float = Field(description="Expected 3-month total return as a decimal, e.g. 0.04")
    thesis: str = Field(description="Two to four sentences: the core investment argument")
    key_drivers: list[str]
    risks: list[str]
    evidence: list[Evidence] = Field(description="Facts from the provided materials that support the thesis")
    used_outside_knowledge: bool = Field(
        description="True if the note relies on anything not in the provided materials (e.g. remembered events)"
    )

    @field_validator("conviction")
    @classmethod
    def _conviction_range(cls, v: float) -> float:
        return min(max(v, 0.0), 1.0)

    @property
    def score(self) -> float:
        """Signed signal in [-1, 1]: rating direction times conviction."""
        return RATING_SCORE[self.rating] * self.conviction
