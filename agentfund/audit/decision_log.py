"""Append-only, hash-chained decision log (JSON Lines).

Each record stores the SHA-256 of the previous record, so editing, deleting or
reordering any past entry breaks :func:`verify`. Publishing the latest hash (for
example in a commit or a tweet) timestamps the whole history up to that point.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

GENESIS = "0" * 64


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _digest(record: dict) -> str:
    body = {k: v for k, v in record.items() if k != "hash"}
    return hashlib.sha256(_canonical(body).encode()).hexdigest()


class DecisionLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq, self._last = 0, GENESIS
        if self.path.exists():
            for rec in self._read():
                self._seq, self._last = rec["seq"] + 1, rec["hash"]

    def _read(self):
        with self.path.open() as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)

    def append(self, kind: str, payload: dict, decision_time=None) -> dict:
        record = {
            "seq": self._seq,
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "decision_time": str(decision_time) if decision_time is not None else None,
            "kind": kind,
            "payload": payload,
            "prev_hash": self._last,
        }
        record["hash"] = _digest(record)
        with self.path.open("a") as fh:
            fh.write(_canonical(record) + "\n")
        self._seq, self._last = self._seq + 1, record["hash"]
        return record

    @property
    def head(self) -> str:
        return self._last

    def records(self) -> list[dict]:
        return list(self._read()) if self.path.exists() else []


def verify(path: str | Path) -> tuple[bool, str]:
    """Check the whole chain. Returns ``(ok, message)``."""
    prev, expected_seq = GENESIS, 0
    with Path(path).open() as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("seq") != expected_seq:
                return False, f"line {n}: sequence {rec.get('seq')} != expected {expected_seq}"
            if rec.get("prev_hash") != prev:
                return False, f"line {n}: prev_hash does not match the previous record"
            if _digest(rec) != rec.get("hash"):
                return False, f"line {n}: record was modified (hash mismatch)"
            prev, expected_seq = rec["hash"], expected_seq + 1
    return True, f"{expected_seq} records verified, head {prev[:16]}"
