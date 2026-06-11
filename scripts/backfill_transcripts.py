"""Backfill historical earnings call transcripts into the FilingStore index.

Sources (see docs/data_source_options.md): the kurry/sp500_earnings_transcripts
HuggingFace dataset (MIT, 2005-2025, S&P 500) or defeatbeta-api. Both are
free; neither package is a project dependency — install on demand:

    pip install datasets            # for the kurry dataset path
    python scripts/backfill_transcripts.py AAPL MSFT NVDA

Each transcript is indexed with its real call date, so backtests get correct
causal isolation. Raw source rows are NOT kept (local storage constraint) —
only the vector index grows, by roughly a few MB per ticker.
"""

import sys

sys.path.insert(0, ".")

from tradingagents.agents.utils.filing_store import get_filing_store  # noqa: E402


def backfill_from_kurry(tickers: list[str]) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("pip install datasets  # required for the kurry dataset path")

    wanted = {t.upper() for t in tickers}
    store = get_filing_store()
    ds = load_dataset("kurry/sp500_earnings_transcripts", split="train", streaming=True)
    count = 0
    for row in ds:
        symbol = str(row.get("symbol", "")).upper()
        if symbol not in wanted:
            continue
        year, quarter = int(row.get("year", 0)), int(row.get("quarter", 0))
        date = str(row.get("date", ""))[:10]
        text = row.get("content") or row.get("transcript") or ""
        if not (year and quarter and date and text):
            continue
        if store.transcript_ingested(symbol, year, quarter):
            continue
        n = store.ingest_transcript(symbol, year, quarter, date, str(text))
        count += 1
        print(f"indexed {symbol} {year}Q{quarter} ({date}): {n} chunks")
    print(f"done: {count} transcripts indexed")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/backfill_transcripts.py TICKER [TICKER ...]")
    backfill_from_kurry(sys.argv[1:])
