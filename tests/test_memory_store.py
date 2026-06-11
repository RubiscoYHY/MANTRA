"""Regression tests for TradingMemoryStore (refactor_plan Phase 1).

These tests encode the DESIRED behavior:
  - semantic search must actually work under the installed chromadb version
    (guards against the silent EmbeddingFunction interface failure)
  - causal isolation: reflections written on day T are invisible on day T
  - cross-ticker (wing) isolation
  - minimum similarity threshold: off-topic queries return nothing
"""

import tempfile

import pytest

from tradingagents.agents.utils.memory_store import (
    TradingMemoryStore,
    build_situation_digest,
)


@pytest.fixture(scope="module")
def store():
    s = TradingMemoryStore(palace_path=tempfile.mkdtemp(), enabled=True)
    s.set_analysis_date("2026-01-10")
    bull_sit = (
        "Market report: AAPL strong upward momentum, RSI 72 overbought, MACD golden cross. "
        "Sentiment: retail bullish on iPhone sales. News: supply chain recovery in Asia. "
        "Fundamentals: P/E 32, growing services revenue."
    )
    s.store_reflection("AAPL", "trader", bull_sit,
                       "Lesson A: do not chase overbought momentum.")
    s.set_analysis_date("2026-01-11")
    bear_sit = (
        "Market report: AAPL declining on weak guidance, RSI 35, death cross forming. "
        "Sentiment: bearish chatter about demand. News: regulatory fine in EU. "
        "Fundamentals: margin compression visible."
    )
    s.store_reflection("AAPL", "trader", bear_sit,
                       "Lesson B: negative guidance plus regulatory news justified early exit.")
    return s


def test_retrieval_returns_hits(store):
    """Search must not fail silently: a semantically close query gets results."""
    store.set_analysis_date("2026-01-13")
    hits = store.retrieve_reflections(
        "AAPL", "trader",
        "AAPL selling off after disappointing guidance, RSI 38, EU antitrust action",
        n_results=2,
    )
    assert hits, "semantic search returned no hits — retrieval is broken"
    assert any("Lesson B" in h["text"] for h in hits)


def test_bearish_query_ranks_bearish_lesson_first(store):
    store.set_analysis_date("2026-01-13")
    hits = store.retrieve_reflections(
        "AAPL", "trader",
        "AAPL selling off after disappointing guidance, RSI 38, EU antitrust action",
        n_results=2,
    )
    assert "Lesson B" in hits[0]["text"]


def test_causal_isolation_same_day(store):
    """A reflection stored on day T must be invisible when querying as-of day T."""
    store.set_analysis_date("2026-01-10")
    hits = store.retrieve_reflections("AAPL", "trader", "anything at all", n_results=5)
    assert hits == []


def test_cross_ticker_isolation(store):
    store.set_analysis_date("2026-01-13")
    hits = store.retrieve_reflections("MSFT", "trader", "guidance miss selloff", n_results=5)
    assert hits == []


def test_similarity_threshold_blocks_offtopic(store):
    """Top-k must not surface semantically unrelated memories."""
    store.set_analysis_date("2026-01-13")
    hits = store.retrieve_reflections(
        "AAPL", "trader",
        "Recipe for chocolate cake with strawberries and cream frosting",
        n_results=2,
    )
    assert hits == [], f"off-topic query should return nothing, got similarities: " \
                       f"{[h['similarity'] for h in hits]}"


def test_disabled_store_is_noop():
    s = TradingMemoryStore(palace_path=tempfile.mkdtemp(), enabled=False)
    s.set_analysis_date("2026-01-10")
    assert s.store_reflection("AAPL", "trader", "sit", "lesson") is False
    assert s.retrieve_reflections("AAPL", "trader", "q") == []


def test_build_situation_digest_truncates():
    long_report = "word " * 5000
    digest = build_situation_digest(long_report, long_report, long_report, long_report)
    assert len(digest) < 4 * 700 + 200
    assert "word" in digest


def test_write_failure_raises_when_enabled():
    """Errors must not be swallowed: invalid date input should raise, not return False."""
    s = TradingMemoryStore(palace_path=tempfile.mkdtemp(), enabled=True)
    with pytest.raises(Exception):
        s.store_reflection("AAPL", "trader", "sit", "lesson")  # no analysis date set
