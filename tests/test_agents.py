import json

import pandas as pd

from agentfund.agents import AnalystAgent, CallLedger, Document, DocumentStore, OfflineLLM, alias_for, build_data_pack
from agentfund.agents.analyst import offline_analyst_responder, render_prompt
from agentfund.agents.schemas import ResearchNote


def _docs():
    return DocumentStore(
        [
            Document("d1", "SYN003", "2018-02-15", "10-K", "SYN003 revenue grew in 2017.\n\nMargins compressed due to competition."),
            Document("d2", "SYN003", "2019-02-15", "10-K", "Synthetic Co 003 expects a weak 2019 outlook and impairment."),
        ]
    )


def test_note_is_structured_and_ledgered(store):
    led = CallLedger()
    agent = AnalystAgent(OfflineLLM(offline_analyst_responder, led), docs=_docs())
    note = agent.research(store.view("2018-06-29"), "SYN003")
    assert isinstance(note, ResearchNote)
    assert note.ticker == "SYN003" and note.as_of == "2018-06-29"
    assert -1 <= note.score <= 1
    assert led.totals()["calls"] == 1


def test_retrieval_respects_filed_date(store):
    pack = build_data_pack(store.view("2018-06-29"), "SYN003", _docs())
    text = json.dumps(pack)
    assert "competition" in text and "impairment" not in text


def test_anonymized_pack_hides_identity_and_dates(store):
    view = store.view("2019-06-28")
    pack = build_data_pack(view, "SYN003", _docs(), anonymize=True)
    blob = render_prompt(pack)
    assert "SYN003" not in blob and "Synthetic Co 003" not in blob
    assert alias_for("SYN003") in blob
    assert "2019" not in blob and "2018" not in blob
    assert pack["price"]["rebased_to_100_one_year_ago"] is True


def test_cache_avoids_repeat_calls(store, tmp_path):
    led = CallLedger()
    agent = AnalystAgent(OfflineLLM(offline_analyst_responder, led), cache_dir=tmp_path)
    v = store.view("2018-06-29")
    a, b = agent.research(v, "SYN004"), agent.research(v, "SYN004")
    assert a == b and led.totals()["calls"] == 1
