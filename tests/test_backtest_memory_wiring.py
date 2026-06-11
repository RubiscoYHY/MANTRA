"""Integration test: backtest-mode memory wiring without LLM/API calls.

Verifies the full write path (report persistence + reflections) and the
next-day read path using a stub LLM, mirroring what reflect_and_remember()
and _persist_daily_reports() do during a real backtest.
"""

import tempfile

import pytest

from tradingagents.agents.utils.memory_store import TradingMemoryStore
from tradingagents.graph.reflection import Reflector
from tradingagents.graph.trading_graph import TradingAgentsGraph


class _StubLLM:
    """Minimal stand-in for a LangChain chat model."""

    class _Msg:
        content = "Stub lesson: trust confirmed breakouts, distrust overbought chases."

    def invoke(self, messages):
        return self._Msg()


class _GraphShell:
    """Carries just the attribute _persist_daily_reports() needs."""

    def __init__(self, store):
        self.memory_store = store


@pytest.fixture()
def store():
    s = TradingMemoryStore(palace_path=tempfile.mkdtemp(), enabled=True)
    s.set_analysis_date("2026-02-02")
    return s


def _fake_state():
    return {
        "company_of_interest": "NVDA",
        "market_report": "NVDA breaking out on strong volume, RSI 65.",
        "sentiment_report": "Retail euphoric about datacenter demand.",
        "news_report": "New GPU export rules announced by regulators.",
        "fundamentals_report": "Revenue growth 60% YoY, gross margin expanding.",
        "investment_debate_state": {
            "bull_history": "Bull: breakout is confirmed by volume.",
            "bear_history": "Bear: export rules are a structural risk.",
            "judge_decision": "Judge: bull case is better evidenced.",
        },
        "trader_investment_plan": "Enter half position, stop below breakout level.",
        "risk_debate_state": {"judge_decision": "Approve half position."},
    }


def test_full_backtest_day_writes_and_next_day_reads(store):
    state = _fake_state()

    # Day T: persist the four analyst reports (central write point).
    TradingAgentsGraph._persist_daily_reports(_GraphShell(store), "NVDA", state)

    # Day T: reflections for all five roles via stub LLM.
    reflector = Reflector(_StubLLM())
    for fn in (
        reflector.reflect_bull_researcher,
        reflector.reflect_bear_researcher,
        reflector.reflect_trader,
        reflector.reflect_invest_judge,
        reflector.reflect_portfolio_manager,
    ):
        fn(state, returns_losses=0.012, memory_store=store)

    assert store.db_writes == 9  # 4 reports + 5 reflections

    # Day T+1: reflections become visible (causal isolation: valid_from=T+1).
    store.set_analysis_date("2026-02-03")
    hits = store.retrieve_reflections(
        "NVDA", "trader",
        "NVDA breakout on volume with regulatory export risk",
        n_results=2,
    )
    assert hits, "trader reflection not retrievable on T+1"
    assert "Stub lesson" in hits[0]["text"]
    assert store.db_reads > 0

    # Same-day sentiment is retrievable for the social analyst (valid_from=T).
    sent = store.retrieve_similar_sentiment(
        "NVDA", "retail investors euphoric about AI datacenter demand", n_results=1
    )
    assert sent and "euphoric" in sent[0]["text"]
