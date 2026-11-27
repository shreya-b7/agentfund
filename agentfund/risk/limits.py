"""Hard risk limits.

Limits are a frozen dataclass with a fingerprint. Agents may *propose* tighter
limits through :meth:`RiskLimits.tightened`, which refuses any change that would
loosen a limit. Nothing in the agent layer can construct looser limits for the
risk officer; the officer is built once with the firm's policy limits.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace

# For each limit: does a *smaller* value mean tighter? (True for caps, False for floors.)
_SMALLER_IS_TIGHTER = {
    "max_position_weight": True,
    "max_gross_exposure": True,
    "max_sector_weight": True,
    "max_turnover": True,
    "max_adv_participation": True,
    "min_price": False,
    "drawdown_kill_switch": True,
    "long_only": None,  # bool: True is tighter
}


@dataclass(frozen=True)
class RiskLimits:
    max_position_weight: float = 0.10  # per name, fraction of equity
    max_gross_exposure: float = 1.00  # sum |w|
    max_sector_weight: float = 0.30  # sum |w| within a sector
    max_turnover: float = 0.60  # one-way turnover per rebalance, sum |dw| / 2
    max_adv_participation: float = 0.05  # trade notional / (20d avg dollar volume)
    min_price: float = 5.0  # no new positions below this price
    drawdown_kill_switch: float = 0.25  # flatten when drawdown from peak exceeds this
    long_only: bool = True

    def __post_init__(self) -> None:
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name != "long_only" and (not isinstance(v, (int, float)) or v < 0):
                raise ValueError(f"{f.name} must be a non-negative number, got {v!r}")

    def fingerprint(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def tightened(self, **changes) -> "RiskLimits":
        """Return a copy with some limits tightened. Raises if any change loosens a limit."""
        for name, new in changes.items():
            if name not in _SMALLER_IS_TIGHTER:
                raise KeyError(f"unknown limit {name!r}")
            old = getattr(self, name)
            rule = _SMALLER_IS_TIGHTER[name]
            loosens = (old and not new) if rule is None else (new > old if rule else new < old)
            if loosens:
                raise PermissionError(f"refusing to loosen {name}: {old!r} -> {new!r}")
        return replace(self, **changes)
