"""Offline tests for the FilingStore (10-K RAG index).

No network: filings are injected via ingest_text() with synthetic content.
The live EDGAR path is covered by the manual smoke script (network, keyless).
"""

import tempfile

import pytest

from tradingagents.agents.utils.filing_store import FilingStore, _chunk
from tradingagents.dataflows.filings import FilingRef, _html_to_text


@pytest.fixture(scope="module")
def store():
    s = FilingStore(index_path=tempfile.mkdtemp())
    ref_old = FilingRef(
        ticker="AAPL", cik=320193, form="10-K",
        accession="000032019323000106", filed_date="2025-11-01",
        primary_doc="aapl.htm",
    )
    text_old = (
        "Item 1A. Risk Factors. The Company depends on component supply from a "
        "small number of suppliers in Asia; disruption of the supply chain "
        "could materially harm results of operations. "
        + "General corporate boilerplate. " * 200
        + "Item 7. Management's Discussion. Services revenue grew due to "
        "advertising, cloud, and payment services."
    )
    s.ingest_text(ref_old, text_old)

    ref_new = FilingRef(
        ticker="AAPL", cik=320193, form="10-Q",
        accession="000032019326000044", filed_date="2026-05-02",
        primary_doc="aapl_q.htm",
    )
    s.ingest_text(ref_new, "Item 1A update: new tariff regulations introduced "
                           "in 2026 create additional import cost risk." )
    return s


def test_search_returns_relevant_chunk(store):
    hits = store.search("AAPL", "supply chain risk from suppliers", as_of="2026-06-01")
    assert hits
    assert "supply" in hits[0]["text"].lower()


def test_causal_isolation_by_filed_date(store):
    """A filing from 2026-05 must be invisible to a 2026-01 backtest day."""
    hits = store.search("AAPL", "tariff regulations import cost", as_of="2026-01-15")
    assert all(h["filed_date"] <= 20260115 for h in hits)

    hits_later = store.search("AAPL", "tariff regulations import cost", as_of="2026-06-01")
    assert any(h["filed_date"] == 20260502 for h in hits_later)


def test_staleness_window_excludes_old_filings(store):
    """A filing older than the validity window (3y default) is filtered
    INSIDE the search, not post-hoc."""
    ancient = FilingRef(
        ticker="AAPL", cik=320193, form="10-K",
        accession="000032019320000010", filed_date="2020-10-30",
        primary_doc="aapl_old.htm",
    )
    store.ingest_text(ancient, "Item 1A. Ancient risk: dependence on CD-ROM "
                               "distribution channels for legacy software.")
    hits = store.search("AAPL", "CD-ROM distribution channel risk", as_of="2026-06-01")
    assert all(h["filed_date"] >= 20230602 for h in hits)
    # The same query with an as_of inside the window DOES see it.
    hits_2022 = store.search("AAPL", "CD-ROM distribution channel risk", as_of="2022-06-01")
    assert any(h["filed_date"] == 20201030 for h in hits_2022)


def test_cross_ticker_isolation(store):
    assert store.search("MSFT", "supply chain", as_of="2026-06-01") == []


def test_reingest_is_idempotent(store):
    assert store._is_ingested("000032019323000106")


def test_transcript_search_is_isolated_from_filing_search(store):
    """Transcripts share the collection but must not pollute 10-K search,
    and vice versa; each has its own validity window."""
    store.ingest_transcript(
        "AAPL", 2026, 1, "2026-02-01",
        "CFO: we expect gross margin between 46 and 47 percent next quarter "
        "driven by services mix.",
    )
    # Filing search must NOT return the call chunk.
    f_hits = store.search("AAPL", "gross margin guidance next quarter", as_of="2026-06-01")
    assert all(h["form"] != "EARNINGS_CALL" for h in f_hits)
    # Transcript search returns it...
    t_hits = store.search_transcripts("AAPL", "gross margin guidance", as_of="2026-06-01")
    assert t_hits and t_hits[0]["form"] == "EARNINGS_CALL"
    # ...but not before the call date (causal isolation)...
    assert store.search_transcripts("AAPL", "gross margin guidance", as_of="2026-01-15") == []
    # ...and not after its 365-day validity expires.
    assert store.search_transcripts("AAPL", "gross margin guidance", as_of="2027-06-01") == []


def test_candidate_quarters():
    from tradingagents.dataflows.transcripts import _candidate_quarters
    assert _candidate_quarters("2026-06-10") == [(2026, 1), (2025, 4), (2025, 3)]
    assert _candidate_quarters("2026-01-05") == [(2025, 4), (2025, 3), (2025, 2)]


def test_chunking_bounds():
    chunks = _chunk("word " * 10000)
    assert chunks
    assert all(len(c) <= 2000 for c in chunks)


def test_html_to_text_strips_ixbrl():
    html = (
        "<html><ix:header><ix:hidden>XBRL JUNK us-gaap:Thing</ix:hidden></ix:header>"
        "<body><p>Real sentence one.</p><script>var x=1;</script>"
        "<div>Real sentence two.</div></body></html>"
    )
    text = _html_to_text(html)
    assert "Real sentence one." in text and "Real sentence two." in text
    assert "XBRL JUNK" not in text and "var x=1" not in text
