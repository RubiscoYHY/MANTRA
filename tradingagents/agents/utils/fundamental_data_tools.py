from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor


@tool
def search_10k(
    query: Annotated[str, "natural-language question about the company's latest 10-K/10-Q, e.g. 'main risk factors related to supply chain'"],
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
) -> str:
    """
    Semantic search inside the company's most recent SEC 10-K (or 10-Q)
    filed on or before curr_date. Sourced from SEC EDGAR (free) and indexed
    locally; only filings that already existed on curr_date are visible.
    Returns the most relevant passages with their filing date.
    """
    from tradingagents.agents.utils.filing_store import get_filing_store
    from tradingagents.observability import get_recorder

    store = get_filing_store()
    try:
        ref = store.ensure_ingested(ticker, curr_date)
    except Exception as exc:
        get_recorder().metric("filing", {"available": False, "error": str(exc)[:200]})
        return f"10-K retrieval unavailable: {exc}"

    if ref is None:
        get_recorder().metric("filing", {"available": False, "error": None})
        return (
            f"No 10-K/10-Q filing exists on SEC EDGAR for {ticker} on or "
            f"before {curr_date} (non-US listings are not covered by EDGAR)."
        )

    hits = store.search(ticker, query, as_of=curr_date, n_results=4)
    sample = (hits[0]["text"][:300] + " …") if hits else None
    get_recorder().metric("filing", {
        "available": True,
        "form": ref.form,
        "filed_date": ref.filed_date,
        "accession": ref.accession,
        "n_hits": len(hits),
        "sample": sample,
    })

    if not hits:
        return (
            f"{ref.form} filed {ref.filed_date} is indexed, but no passage "
            f"matched the query: {query!r}. Try different wording."
        )

    parts = [
        f"Passages from {ticker} {ref.form} (filed {ref.filed_date}, "
        f"accession {ref.accession}), most relevant first:\n"
    ]
    for h in hits:
        parts.append(f"[similarity {h['similarity']}]\n{h['text']}\n")
    return "\n---\n".join(parts)


@tool
def search_earnings_call(
    query: Annotated[str, "natural-language question about the latest earnings call, e.g. 'management guidance on margins for next quarter'"],
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
) -> str:
    """
    Semantic search inside the company's most recent quarterly earnings call
    transcript held on or before curr_date (verbatim management remarks and
    Q&A). Only calls that already happened by curr_date are visible.
    Returns the most relevant passages with the call date.
    """
    from tradingagents.agents.utils.filing_store import get_filing_store
    from tradingagents.observability import get_recorder

    store = get_filing_store()
    try:
        ref = store.ensure_transcript(ticker, curr_date)
    except Exception as exc:
        get_recorder().metric("earnings_call", {"available": False, "error": str(exc)[:200]})
        return f"Earnings call retrieval unavailable: {exc}"

    hits = store.search_transcripts(ticker, query, as_of=curr_date, n_results=4)
    sample = (hits[0]["text"][:300] + " …") if hits else None
    get_recorder().metric("earnings_call", {
        "available": bool(hits),
        "quarter": f"{ref.year}Q{ref.quarter}" if ref else None,
        "event_date": ref.event_date if ref else None,
        "n_hits": len(hits),
        "sample": sample,
    })

    if not hits:
        if ref is None:
            return (
                f"No earnings call transcript is available for {ticker} on or "
                f"before {curr_date} (set API_NINJAS_KEY to enable fetching, "
                f"or the company has no recent call)."
            )
        return (
            f"Transcript {ref.year}Q{ref.quarter} (call date {ref.event_date}) "
            f"is indexed, but no passage matched: {query!r}. Try different wording."
        )

    header = f"Passages from {ticker} earnings call"
    if ref:
        header += f" {ref.year}Q{ref.quarter} (call date {ref.event_date})"
    parts = [header + ", most relevant first:\n"]
    for h in hits:
        parts.append(f"[similarity {h['similarity']}]\n{h['text']}\n")
    return "\n---\n".join(parts)


@tool
def get_fundamentals(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
) -> str:
    """
    Retrieve comprehensive fundamental data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing comprehensive fundamental data
    """
    return route_to_vendor("get_fundamentals", ticker, curr_date)


@tool
def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Retrieve balance sheet data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        freq (str): Reporting frequency: annual/quarterly (default quarterly)
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing balance sheet data
    """
    return route_to_vendor("get_balance_sheet", ticker, freq, curr_date)


@tool
def get_cashflow(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Retrieve cash flow statement data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        freq (str): Reporting frequency: annual/quarterly (default quarterly)
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing cash flow statement data
    """
    return route_to_vendor("get_cashflow", ticker, freq, curr_date)


@tool
def get_income_statement(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Retrieve income statement data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        freq (str): Reporting frequency: annual/quarterly (default quarterly)
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing income statement data
    """
    return route_to_vendor("get_income_statement", ticker, freq, curr_date)