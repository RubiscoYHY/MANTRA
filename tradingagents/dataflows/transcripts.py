"""Earnings call transcript access (API Ninjas free tier).

Source choice (researched 2026-06): API Ninjas serves full verbatim
transcripts as JSON, free tier covers recent quarters (~10k req/month);
historical backfill (2005+) comes from the defeatbeta / kurry HuggingFace
datasets via scripts/backfill_transcripts.py, not from this module.

Causal isolation: a transcript only becomes visible from its call date. When
the API does not return an explicit date, we use a CONSERVATIVE estimate of
quarter_end + 45 days — later than virtually all real calls, so a backtest
can never see a transcript before it actually existed (it may see it slightly
late, which is the safe direction).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_API_URL = "https://api.api-ninjas.com/v1/earningscalltranscript"
_TIMEOUT = 30


@dataclass
class TranscriptRef:
    ticker: str
    year: int
    quarter: int
    event_date: str      # YYYY-MM-DD; conservative estimate when unknown
    date_estimated: bool


def _api_key() -> Optional[str]:
    from .config import get_config
    return get_config().get("api_ninjas_key") or os.getenv("API_NINJAS_KEY")


def _quarter_end(year: int, quarter: int) -> date:
    month = quarter * 3
    if month == 12:
        return date(year, 12, 31)
    return date(year, month + 1, 1) - timedelta(days=1)


def _candidate_quarters(as_of: str, n: int = 3) -> list[tuple[int, int]]:
    """Most recent n calendar quarters whose call could exist by as_of.

    A call for quarter Q typically happens 2-6 weeks after the quarter ends,
    so the current (incomplete) quarter is skipped.
    """
    d = datetime.fromisoformat(as_of).date()
    year, q = d.year, (d.month - 1) // 3 + 1
    out = []
    for _ in range(n):
        q -= 1
        if q == 0:
            year, q = year - 1, 4
        out.append((year, q))
    return out


def fetch_transcript(
    ticker: str,
    year: int,
    quarter: int,
    cache_dir: str = "./data/transcripts_cache",
) -> Optional[tuple[TranscriptRef, str]]:
    """Fetch one transcript (disk-cached). Returns (ref, full_text) or None."""
    cache = Path(cache_dir) / f"{ticker.upper()}_{year}Q{quarter}.json"
    data = None
    if cache.exists():
        import json
        data = json.loads(cache.read_text(encoding="utf-8"))
    else:
        key = _api_key()
        if not key:
            logger.info("API_NINJAS_KEY not set; earnings call fetch skipped")
            return None
        resp = requests.get(
            _API_URL,
            params={"ticker": ticker.upper(), "year": year, "quarter": quarter},
            headers={"X-Api-Key": key},
            timeout=_TIMEOUT,
        )
        if resp.status_code != 200:
            logger.warning("Transcript API %s for %s %sQ%s",
                           resp.status_code, ticker, year, quarter)
            return None
        data = resp.json()
        if isinstance(data, list):
            data = data[0] if data else None
        if not data or not data.get("transcript"):
            return None
        import json
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    if not data or not data.get("transcript"):
        return None

    raw_date = str(data.get("date") or "")[:10]
    try:
        event_date, estimated = (
            datetime.fromisoformat(raw_date).strftime("%Y-%m-%d"), False
        )
    except ValueError:
        event_date = (_quarter_end(year, quarter) + timedelta(days=45)).isoformat()
        estimated = True

    ref = TranscriptRef(
        ticker=ticker.upper(), year=year, quarter=quarter,
        event_date=event_date, date_estimated=estimated,
    )
    return ref, str(data["transcript"])


def fetch_latest_transcript(
    ticker: str, as_of: str
) -> Optional[tuple[TranscriptRef, str]]:
    """Most recent transcript whose call date is on or before as_of."""
    for year, quarter in _candidate_quarters(as_of):
        got = fetch_transcript(ticker, year, quarter)
        if got and got[0].event_date <= as_of:
            return got
    return None
