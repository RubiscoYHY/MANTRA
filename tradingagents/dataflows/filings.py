"""SEC EDGAR filing access — free, no API key.

The only requirement is a User-Agent header identifying the requester
(SEC blocks anonymous clients with 403) and staying under 10 req/sec.

Pipeline:
    ticker → CIK (company_tickers.json, cached per process)
         → submissions JSON → latest 10-K/10-Q with filed_date <= as_of
         → primary document HTML → strip iXBRL header + tags → plain text
Downloaded text is cached on disk under data/filings_cache/, so each filing
is fetched from EDGAR exactly once.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"

_MIN_INTERVAL = 0.12  # ~8 req/sec, under SEC's 10 req/sec limit
_last_request_ts = 0.0

_cik_map: Optional[dict[str, int]] = None


def _user_agent() -> str:
    from .config import get_config
    ua = get_config().get("edgar_user_agent") or os.getenv("EDGAR_USER_AGENT")
    return ua or "Mantra research agent rubisco.yau@gmail.com"


def _get(url: str) -> requests.Response:
    """Rate-limited GET with the SEC-required User-Agent."""
    global _last_request_ts
    wait = _MIN_INTERVAL - (time.monotonic() - _last_request_ts)
    if wait > 0:
        time.sleep(wait)
    resp = requests.get(url, headers={"User-Agent": _user_agent()}, timeout=30)
    _last_request_ts = time.monotonic()
    resp.raise_for_status()
    return resp


def ticker_to_cik(ticker: str) -> Optional[int]:
    """Resolve a ticker to its SEC CIK number (None for non-US/unknown)."""
    global _cik_map
    if _cik_map is None:
        data = _get(_TICKERS_URL).json()
        _cik_map = {v["ticker"].upper(): int(v["cik_str"]) for v in data.values()}
    return _cik_map.get(ticker.upper())


@dataclass
class FilingRef:
    ticker: str
    cik: int
    form: str           # "10-K" or "10-Q"
    accession: str      # no dashes
    filed_date: str     # YYYY-MM-DD
    primary_doc: str


def find_latest_filing(
    ticker: str,
    as_of: str,
    forms: tuple[str, ...] = ("10-K",),
) -> Optional[FilingRef]:
    """Most recent filing of the given form(s) with filed_date <= as_of.

    The filed-date bound is the causal isolation gate: a backtest on date T
    must only ever see filings that existed on T.
    """
    cik = ticker_to_cik(ticker)
    if cik is None:
        return None
    subs = _get(_SUBMISSIONS_URL.format(cik=cik)).json()
    recent = subs.get("filings", {}).get("recent", {})
    best: Optional[FilingRef] = None
    for form, filed, acc, doc in zip(
        recent.get("form", []),
        recent.get("filingDate", []),
        recent.get("accessionNumber", []),
        recent.get("primaryDocument", []),
    ):
        if form in forms and filed <= as_of:
            if best is None or filed > best.filed_date:
                best = FilingRef(
                    ticker=ticker.upper(), cik=cik, form=form,
                    accession=acc.replace("-", ""),
                    filed_date=filed, primary_doc=doc,
                )
    return best


_IX_HEADER_RE = re.compile(r"(?s)<ix:header.*?</ix:header>", re.IGNORECASE)
_SCRIPT_STYLE_RE = re.compile(r"(?s)<(script|style)[^>]*>.*?</\1>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\xa0]+")
_NL_RE = re.compile(r"\n{3,}")


def _html_to_text(html: str) -> str:
    """Strip iXBRL header, scripts/styles, and tags; normalize whitespace."""
    html = _IX_HEADER_RE.sub(" ", html)
    html = _SCRIPT_STYLE_RE.sub(" ", html)
    # Preserve block boundaries so chunking respects paragraphs.
    html = re.sub(r"(?i)</(p|div|tr|table|h[1-6]|li)>", "\n", html)
    text = _TAG_RE.sub(" ", html)
    import html as _html
    text = _html.unescape(text)
    text = _WS_RE.sub(" ", text)
    return _NL_RE.sub("\n\n", text).strip()


def fetch_filing_text(ref: FilingRef, cache_dir: str = "./data/filings_cache") -> str:
    """Download (or load from disk cache) the plain text of a filing."""
    cache = Path(cache_dir) / f"{ref.ticker}_{ref.form}_{ref.accession}.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    url = _ARCHIVE_URL.format(cik=ref.cik, acc=ref.accession, doc=ref.primary_doc)
    logger.info("Downloading %s %s for %s (%s)", ref.form, ref.filed_date, ref.ticker, url)
    text = _html_to_text(_get(url).text)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return text
