"""Tests for the run recorder, metrics, and replay cache."""

import json
import tempfile
import threading

from tradingagents.observability import (
    RunRecorder,
    config_digest,
    count_news_items,
    find_replayable,
    get_recorder,
    set_recorder,
)


def _mk(tmp, **kw):
    return RunRecorder(root=tmp, ticker="AAPL", date="2026-06-10",
                       digest="d1", **kw)


def test_record_lifecycle_and_persistence():
    tmp = tempfile.mkdtemp()
    rec = _mk(tmp)
    rec.agent_start("Market Analyst", inputs={"trade_date": "2026-06-10"})
    rec.llm_turn("Market Analyst", "thinking...", ["get_stock_data"])
    rec.tool_call("Market Analyst", "get_stock_data", {"symbol": "AAPL"}, "csv,data")
    rec.agent_finish("Market Analyst", output={"market_report": "Strong uptrend."})
    rec.memory_event("Trader", "reflections_trader", "query text",
                     [{"similarity": 0.7, "trade_date": "2026-06-09", "text": "lesson"}])
    rec.metric("social", {"status": "ok", "fetched": 10, "on_target": 4, "snr": 0.4})
    rec.bump_metric("news", "get_news", 7)
    rec.bump_metric("news", "get_news", 3)
    rec.finish(signal={"signal": "HOLD"}, reports={"market_report": "Strong uptrend."},
               decision_text="HOLD because ...")

    data = json.loads(rec.path.read_text())
    agent = data["agents"]["Market Analyst"]
    assert agent["status"] == "completed"
    assert agent["llm_turns"][0]["tool_calls"] == ["get_stock_data"]
    assert agent["tool_calls"][0]["result_text"] == "csv,data"
    assert data["metrics"]["news"]["get_news"] == 10
    assert data["metrics"]["social"]["snr"] == 0.4
    assert data["memory_events"][0]["agent"] == "Trader"
    assert data["status"] == "completed"


def test_replay_requires_matching_digest():
    tmp = tempfile.mkdtemp()
    rec = _mk(tmp)
    rec.finish(signal={"signal": "BUY"}, reports={}, decision_text="x")
    assert find_replayable(tmp, "AAPL", "2026-06-10", "d1") is not None
    assert find_replayable(tmp, "AAPL", "2026-06-10", "OTHER") is None
    assert find_replayable(tmp, "MSFT", "2026-06-10", "d1") is None


def test_incomplete_record_not_replayable():
    tmp = tempfile.mkdtemp()
    _mk(tmp)  # started but never finished
    assert find_replayable(tmp, "AAPL", "2026-06-10", "d1") is None


def test_null_recorder_is_safe_noop():
    set_recorder(None)
    rec = get_recorder()
    rec.agent_start("X")
    rec.metric("social", {})
    rec.finish({}, {}, "")  # must not raise or write anything


def test_config_digest_stability():
    cfg = {"deep_think_llm": "a", "quick_think_llm": "b", "data_vendors": {"n": 1}}
    assert config_digest(cfg) == config_digest(dict(cfg))
    assert config_digest(cfg) != config_digest({**cfg, "deep_think_llm": "z"})


def test_count_news_items():
    md = "## News\n\n### A (src)\nx\n### B (src)\ny\n"
    assert count_news_items(md) == 2
    av = json.dumps({"items": "2", "feed": [{"title": "a"}, {"title": "b"}]})
    assert count_news_items(av) == 2
    assert count_news_items("") == 0


def test_thread_safety_of_bump():
    tmp = tempfile.mkdtemp()
    rec = _mk(tmp)
    threads = [
        threading.Thread(target=lambda: [rec.bump_metric("news", "k", 1) for _ in range(20)])
        for _ in range(4)
    ]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert rec.get_metrics()["news"]["k"] == 80
