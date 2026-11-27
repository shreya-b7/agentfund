import json

from agentfund.audit import DecisionLog, verify


def test_chain_verifies_and_detects_tampering(tmp_path):
    p = tmp_path / "log.jsonl"
    log = DecisionLog(p)
    for i in range(5):
        log.append("risk_review", {"i": i, "w": {"A": 0.1 * i}})
    assert verify(p)[0]

    # Reopening continues the chain.
    DecisionLog(p).append("note", {"x": 1})
    assert verify(p) == (True, verify(p)[1]) and "6 records" in verify(p)[1]

    lines = p.read_text().splitlines()
    rec = json.loads(lines[2])
    rec["payload"]["w"]["A"] = 9.9
    lines[2] = json.dumps(rec)
    p.write_text("\n".join(lines) + "\n")
    ok, msg = verify(p)
    assert not ok and "line 3" in msg


def test_deleting_a_record_is_detected(tmp_path):
    p = tmp_path / "log.jsonl"
    log = DecisionLog(p)
    for i in range(3):
        log.append("k", {"i": i})
    lines = p.read_text().splitlines()
    p.write_text("\n".join([lines[0], lines[2]]) + "\n")
    assert not verify(p)[0]
