"""
social_data_tools.py

Provides two LangChain tools for live social media data collection, plus a
FinBERT-based aggregation function that compresses raw posts into a compact
structured summary before passing to the social media analyst LLM.

Architecture
------------
  _fetch_stocktwits_raw()  ─┐
  _fetch_reddit_raw()      ─┴─ finbert_aggregate() ──► ~600-token summary
                                                              │
                                               injected into analyst prompt

The @tool-decorated functions are thin wrappers for potential standalone /
LangGraph tool-call use. The analyst node calls the internal helpers directly
so it has structured post dicts for FinBERT without parsing string output.

FinBERT dependency
------------------
Requires:  pip install transformers torch
Model is cached by HuggingFace at ~/.cache/huggingface/hub/ on first run.
Pre-download (optional, avoids delay on first analysis):
    python -c "from transformers import pipeline; pipeline('text-classification', model='ProsusAI/finbert')"

If transformers is not installed, finbert_aggregate() falls back to a plain
formatted post listing so the analyst can still run without crashing.
"""

import logging
import re
import time
import requests
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Annotated, Optional
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

try:
    from transformers import pipeline as _hf_pipeline
    _FINBERT_AVAILABLE = True
except ImportError:
    _FINBERT_AVAILABLE = False


def _detect_device() -> str:
    """Return the best available torch device: cuda, mps, or cpu."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"

# Optional callback invoked with "in_progress" / "completed" around FinBERT inference.
# Registered externally (e.g. CLI) via set_finbert_status_callback().
_finbert_status_callback = None


def set_finbert_status_callback(callback) -> None:
    """Register a callback(status: str) to receive FinBERT progress events."""
    global _finbert_status_callback
    _finbert_status_callback = callback


def _notify_finbert_status(status: str) -> None:
    if _finbert_status_callback is not None:
        _finbert_status_callback(status)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
_REQUEST_TIMEOUT = 10
_REDDIT_COOLDOWN = 3        # seconds between subreddit requests
_REDDIT_SUBREDDITS = ["wallstreetbets", "stocks", "investing"]
_FINBERT_TEXT_LIMIT = 512   # safe char limit for BERT's 512-token max
_SOCIAL_RECENCY_DAYS = 3    # posts older than this are discarded before FinBERT

# Module-level cache: keyed by ticker so each analysis run fetches once.
# The analyst node may be re-entered 2-3 times within one propagate() call;
# this avoids redundant network requests on those subsequent entries.
_posts_cache: dict[str, list[dict]] = {}


# ---------------------------------------------------------------------------
# Internal fetch helpers (return structured dicts, not formatted strings)
# ---------------------------------------------------------------------------

def _fetch_stocktwits_raw(ticker: str, max_pages: int = 10) -> list[dict]:
    """
    Fetch StockTwits messages for the last _SOCIAL_RECENCY_DAYS days.

    Paginates backwards using the cursor.max message-ID until the oldest post
    in a batch predates the recency window or there are no more pages.
    Each page returns up to 30 messages (StockTwits API hard limit).
    """
    base_url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    cutoff = datetime.now(timezone.utc) - timedelta(days=_SOCIAL_RECENCY_DAYS)

    posts: list[dict] = []
    params: dict = {"limit": 30}

    for _ in range(max_pages):
        try:
            resp = requests.get(
                base_url, params=params,
                headers=_HEADERS, timeout=_REQUEST_TIMEOUT,
            )
        except requests.RequestException:
            break

        if resp.status_code == 429:
            time.sleep(_REDDIT_COOLDOWN * 3)
            break
        if resp.status_code != 200:
            break

        try:
            data = resp.json()
            messages = data.get("messages", [])
            cursor  = data.get("cursor", {})
        except ValueError:
            break

        if not messages:
            break

        oldest_in_batch: datetime | None = None
        for msg in messages:
            body = msg.get("body", "").replace("\n", " ").strip()
            if not body:
                continue
            ts_str = msg.get("created_at", "")
            try:
                post_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if post_dt.tzinfo is None:
                    post_dt = post_dt.replace(tzinfo=timezone.utc)
            except (ValueError, AttributeError):
                post_dt = None

            # Track the oldest timestamp in this batch for the exit check
            if post_dt is not None:
                if oldest_in_batch is None or post_dt < oldest_in_batch:
                    oldest_in_batch = post_dt

            # Only collect posts within the recency window
            if post_dt is None or post_dt >= cutoff:
                sentiment_obj = msg.get("entities", {}).get("sentiment")
                native_sentiment = sentiment_obj.get("basic") if sentiment_obj else None
                posts.append({
                    "text": body,
                    "source": "StockTwits",
                    "score": msg.get("likes", {}).get("total", 0),
                    "timestamp": ts_str,
                    "native_sentiment": native_sentiment,   # bullish/bearish/None
                })

        # Stop if we've gone past the recency window or no more pages
        if oldest_in_batch is not None and oldest_in_batch < cutoff:
            break
        if not cursor.get("more"):
            break

        # Advance cursor backwards in time
        params = {"limit": 30, "max": cursor["max"]}

    return posts


def _fetch_reddit_raw(ticker: str, limit: int = 15) -> list[dict]:
    """Fetch Reddit posts from WSB/stocks/investing. Returns list of post dicts."""
    posts = []
    per_sub = min(limit, 25)

    for subreddit in _REDDIT_SUBREDDITS:
        url = (
            f"https://old.reddit.com/r/{subreddit}/search.json"
            f"?q={ticker}&sort=new&restrict_sr=on&limit={per_sub}"
        )
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT)
        except requests.RequestException:
            time.sleep(_REDDIT_COOLDOWN)
            continue

        if resp.status_code == 429:
            time.sleep(_REDDIT_COOLDOWN * 3)
            continue
        if resp.status_code != 200:
            time.sleep(_REDDIT_COOLDOWN)
            continue

        try:
            children = resp.json()["data"]["children"]
        except (ValueError, KeyError):
            time.sleep(_REDDIT_COOLDOWN)
            continue

        for child in children:
            post = child.get("data", {})
            title = post.get("title", "").strip()
            body = post.get("selftext", "").replace("\n", " ").strip()
            # Combine title + body excerpt for richer signal
            text = (title + " " + body[:300]).strip() if body else title
            if not text:
                continue
            posts.append({
                "text": text,
                "source": f"r/{subreddit}",
                "score": post.get("score", 0),
                "timestamp": post.get("created_utc", ""),
                "native_sentiment": None,
            })

        time.sleep(_REDDIT_COOLDOWN)

    return posts


def _parse_post_dt(ts) -> Optional[datetime]:
    """Parse a post timestamp (Unix epoch float or ISO-8601 string) to aware UTC."""
    if not ts:
        return None
    try:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, OSError, OverflowError):
        return None


def _filter_recent_posts(
    posts: list[dict],
    max_age_days: int = _SOCIAL_RECENCY_DAYS,
    trade_date: Optional[str] = None,
) -> list[dict]:
    """
    Keep posts inside the window [reference - max_age_days, reference].

    Args:
        trade_date: ISO date of the analysis day. When given, it is the
            reference point — posts published AFTER it are future data and are
            dropped (look-ahead guard), and posts with unparseable timestamps
            are dropped too (causality cannot be verified). When None, the
            reference is "now" and unparseable timestamps get the benefit of
            the doubt.
    """
    if trade_date:
        # End of the trade day, UTC.
        reference = datetime.fromisoformat(trade_date).replace(
            tzinfo=timezone.utc
        ) + timedelta(days=1)
        keep_unparseable = False
    else:
        reference = datetime.now(timezone.utc)
        keep_unparseable = True
    cutoff = reference - timedelta(days=max_age_days)

    kept = []
    for p in posts:
        post_dt = _parse_post_dt(p.get("timestamp", ""))
        if post_dt is None:
            if keep_unparseable:
                kept.append(p)
            continue
        if cutoff <= post_dt <= reference:
            kept.append(p)
    return kept


def social_data_status(trade_date: Optional[str]) -> str:
    """Return "live" when trade_date is within the fetchable window, else "historical".

    Public StockTwits/Reddit endpoints only expose recent posts; for an
    analysis date older than the recency window no on-date data can exist.
    """
    if not trade_date:
        return "live"
    try:
        td = datetime.fromisoformat(trade_date).replace(tzinfo=timezone.utc)
    except ValueError:
        return "live"
    age = datetime.now(timezone.utc) - td
    return "live" if age <= timedelta(days=_SOCIAL_RECENCY_DAYS) else "historical"


def get_social_posts_cached(ticker: str, trade_date: Optional[str] = None) -> list[dict]:
    """
    Return combined StockTwits + Reddit posts for a ticker, filtered to the
    recency window ending at trade_date (or now). Cached per (ticker, date)
    to avoid redundant fetches within one analysis run.

    Backtest behavior: for a historical trade_date the public APIs cannot
    return on-date posts, so NO network call is made and an empty list is
    returned — the caller must report social data as missing, not neutral.
    """
    key = f"{ticker}|{trade_date or 'live'}"
    if key not in _posts_cache:
        if social_data_status(trade_date) == "historical":
            logger.info(
                "Social data unavailable for %s on %s (historical date; "
                "public APIs expose recent posts only)", ticker, trade_date,
            )
            _posts_cache[key] = []
        else:
            raw = _fetch_stocktwits_raw(ticker) + _fetch_reddit_raw(ticker)
            from tradingagents.dataflows.archiver import get_archiver
            get_archiver().archive(
                raw, source="social",
                fetched_at=datetime.now(timezone.utc).isoformat(),
            )
            _posts_cache[key] = _filter_recent_posts(raw, trade_date=trade_date)
    return _posts_cache[key]


def clear_posts_cache() -> None:
    """Clear the posts cache. Call between analysis runs for fresh data."""
    _posts_cache.clear()


# ---------------------------------------------------------------------------
# Subject filtering — drop posts whose primary subject is NOT the ticker
# ---------------------------------------------------------------------------
# Keyword search ("q=AAPL") matches any passing mention, and FinBERT only
# classifies sentiment — it has no notion of WHICH company the sentiment is
# about. These filters run BEFORE FinBERT so the aggregate distribution is
# computed over on-target posts only.

# Uppercase tokens that look like tickers but are common slang/abbreviations.
_TICKER_STOPWORDS = frozenset({
    "A", "I", "DD", "THE", "CEO", "CFO", "US", "USA", "ETF", "IPO", "AI",
    "EPS", "PE", "ATH", "YOLO", "FOMO", "IMO", "TLDR", "EDIT", "OP", "WSB",
    "FED", "SEC", "GDP", "CPI", "EOD", "AH", "PM", "IV", "OTM", "ITM", "LOL",
    "TA", "PT", "EOY", "RSI", "MACD", "API", "OK", "NOT", "ALL", "BUY",
    "SELL", "HOLD", "CALL", "PUT", "PUTS", "IT", "ON", "TO", "BE", "OR",
})

_TICKER_TOKEN_RE = re.compile(r"\$([A-Za-z]{1,5})\b|\b([A-Z]{2,5})\b")


def _ticker_mention_counts(text: str) -> dict[str, int]:
    """Count cashtag/uppercase-token mentions that plausibly refer to tickers."""
    counts: dict[str, int] = {}
    for m in _TICKER_TOKEN_RE.finditer(text):
        cashtag, bare = m.group(1), m.group(2)
        token = (cashtag or bare).upper()
        if cashtag is None and token in _TICKER_STOPWORDS:
            continue
        counts[token] = counts.get(token, 0) + 1
    return counts


def subject_prefilter(posts: list[dict], ticker: str) -> list[dict]:
    """
    Rule-based primary-subject filter (zero cost, runs before FinBERT).

    A post passes if ANY of:
      1. it contains the exact cashtag ($TICKER);
      2. the ticker appears in the post title;
      3. the ticker is the most-mentioned ticker-like token in the text.
    """
    target = ticker.upper()
    kept = []
    for p in posts:
        text = p.get("text", "") or ""
        title = p.get("title", "") or ""

        if re.search(rf"\${target}\b", text, flags=re.IGNORECASE) or \
           re.search(rf"\${target}\b", title, flags=re.IGNORECASE):
            kept.append(p)
            continue

        if re.search(rf"\b{target}\b", title, flags=re.IGNORECASE):
            kept.append(p)
            continue

        counts = _ticker_mention_counts(text)
        own = counts.pop(target, 0)
        if own > 0 and own >= max(counts.values(), default=0):
            kept.append(p)
    return kept


@lru_cache(maxsize=1)
def _get_nli_pipeline(model_name: str):
    """Lazy-load the zero-shot NLI pipeline. Cached after first call."""
    return _hf_pipeline(
        "zero-shot-classification",
        model=model_name,
        device=_detect_device(),
    )


def nli_subject_filter(
    posts: list[dict],
    ticker: str,
    threshold: float = 0.6,
) -> list[dict]:
    """
    Second-stage subject filter: zero-shot NLI entailment check that the post
    is primarily about the target ticker. Config-gated via
    ``social_nli_filter`` (model name in ``social_nli_model``).

    Falls back to returning posts unchanged when transformers is missing or
    the model fails — the rule prefilter has already run at that point.
    """
    from tradingagents.dataflows.config import get_config
    cfg = get_config()
    if not cfg.get("social_nli_filter", True) or not posts or not _FINBERT_AVAILABLE:
        return posts

    model_name = cfg.get("social_nli_model", "typeform/distilbert-base-uncased-mnli")
    hypothesis_label = f"about {ticker.upper()} stock"
    try:
        pipe = _get_nli_pipeline(model_name)
        texts = [p.get("text", "")[:_FINBERT_TEXT_LIMIT] for p in posts]
        results = pipe(
            texts,
            candidate_labels=[hypothesis_label, "about a different company or topic"],
            batch_size=16,
        )
        if isinstance(results, dict):
            results = [results]
        kept = []
        for post, res in zip(posts, results):
            scores = dict(zip(res["labels"], res["scores"]))
            if scores.get(hypothesis_label, 0.0) >= threshold:
                kept.append(post)
        return kept
    except Exception as exc:
        logger.warning("NLI subject filter failed (%s); passing posts through", exc)
        return posts


# ---------------------------------------------------------------------------
# LangChain tools (thin wrappers for standalone / tool-call use)
# ---------------------------------------------------------------------------

def _format_posts_as_string(posts: list[dict], ticker: str, source_label: str) -> str:
    if not posts:
        return f"[{source_label}] No posts found for {ticker}."
    lines = [f"{source_label} posts for ${ticker}:\n"]
    for p in posts:
        lines.append(
            f"[{p['timestamp']}] [score:{p['score']}] "
            f"[{p.get('native_sentiment') or 'unlabeled'}] {p['text']}"
        )
    return "\n".join(lines)


@tool
def get_stocktwits_stream(
    ticker: Annotated[str, "Stock ticker symbol, e.g. AAPL"],
    limit: Annotated[int, "Number of messages to fetch (max 30)"] = 30,
) -> str:
    """
    Fetch recent StockTwits messages for a ticker symbol using the public
    StockTwits stream API. No API key required. Rate limit: 200 req/hour.
    Returns a formatted string with message body, sentiment label, and timestamp.
    """
    posts = _fetch_stocktwits_raw(ticker, limit)
    return _format_posts_as_string(posts, ticker, "StockTwits")


@tool
def get_reddit_posts(
    ticker: Annotated[str, "Stock ticker symbol, e.g. AAPL"],
    limit: Annotated[int, "Number of posts per subreddit (max 25)"] = 15,
) -> str:
    """
    Fetch recent Reddit posts mentioning a stock ticker from
    r/wallstreetbets, r/stocks, and r/investing via Reddit's public
    JSON search endpoint. No API key required.
    Returns a formatted string with post title, body excerpt, score, and timestamp.
    """
    posts = _fetch_reddit_raw(ticker, limit)
    return _format_posts_as_string(posts, ticker, "Reddit")


# ---------------------------------------------------------------------------
# FinBERT aggregation
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_finbert_pipeline():
    """Lazy-load FinBERT on the best available device. Cached after first call."""
    return _hf_pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        top_k=None,         # return scores for all three labels
        device=_detect_device(),
    )


def finbert_aggregate(
    posts: list[dict],
    min_neutral_confidence: float = 0.65,
    top_n: int = 5,
    total_fetched: Optional[int] = None,
) -> str:
    """
    Run FinBERT on a list of post dicts and return a compact structured summary.

    Input post dict keys:  text, source, score (upvotes), timestamp
    Output: ~600-token string with sentiment distribution + top posts per class.

    Falls back to a plain post listing if transformers is not installed.

    Args:
        posts: List of post dicts (already subject-filtered).
        min_neutral_confidence: Neutral posts below this confidence are dropped
                                as low-signal noise.
        top_n: Number of top posts to surface per sentiment class.
        total_fetched: Post count BEFORE subject filtering. When provided, the
            summary reports both numbers so the analyst LLM can judge sample
            size and noise level.
    """
    if not posts:
        if total_fetched:
            return (
                f"All {total_fetched} fetched social media posts were filtered "
                f"out as off-target (their primary subject was a different "
                f"company or topic). No on-target social sentiment is available."
            )
        return "No social media posts available for this ticker."

    _notify_finbert_status("in_progress")

    try:
        return _finbert_aggregate_inner(posts, min_neutral_confidence, top_n, total_fetched)
    finally:
        _notify_finbert_status("completed")


def _finbert_aggregate_inner(
    posts: list[dict],
    min_neutral_confidence: float,
    top_n: int,
    total_fetched: Optional[int] = None,
) -> str:
    if not _FINBERT_AVAILABLE:
        # Graceful fallback: plain listing, no FinBERT
        lines = ["[FinBERT not installed — raw posts listed without sentiment scoring]\n"]
        for p in posts[:20]:
            lines.append(f"[{p['source']}] [score:{p['score']}] {p['text'][:200]}")
        return "\n".join(lines)

    pipe = _get_finbert_pipeline()

    # Prepare texts (truncate to BERT safe limit)
    valid_posts = [p for p in posts if p.get("text", "").strip()]
    texts = [p["text"][:_FINBERT_TEXT_LIMIT] for p in valid_posts]

    # Batch inference
    raw_results = pipe(texts, batch_size=32, truncation=True, max_length=512)

    # Attach FinBERT labels and confidence to each post
    annotated: list[dict] = []
    for post, label_scores in zip(valid_posts, raw_results):
        top = max(label_scores, key=lambda x: x["score"])
        post = dict(post)   # copy to avoid mutating cache
        post["finbert_label"] = top["label"]        # positive/negative/neutral
        post["finbert_conf"] = round(top["score"], 3)
        annotated.append(post)

    # Filter low-confidence neutral noise
    filtered = [
        p for p in annotated
        if not (p["finbert_label"] == "neutral" and p["finbert_conf"] < min_neutral_confidence)
    ]

    if not filtered:
        filtered = annotated  # keep everything if filter removes all

    # Group by label
    bullish = [p for p in filtered if p["finbert_label"] == "positive"]
    bearish = [p for p in filtered if p["finbert_label"] == "negative"]
    neutral = [p for p in filtered if p["finbert_label"] == "neutral"]

    # Rank by upvotes × confidence (surfaces high-engagement, high-certainty posts)
    def _rank(p: dict) -> float:
        return p.get("score", 0) * p["finbert_conf"]

    bullish.sort(key=_rank, reverse=True)
    bearish.sort(key=_rank, reverse=True)

    total = len(filtered)
    sample_note = (
        f"Posts fetched: {total_fetched}, on-target after subject filtering: "
        f"{len(posts)}, scored: {total}"
        if total_fetched is not None
        else f"Total posts after filtering: {total}"
    )
    lines = [
        "=== Social Media Sentiment (FinBERT) ===",
        f"Distribution — Bullish: {len(bullish)/total:.0%} ({len(bullish)}), "
        f"Bearish: {len(bearish)/total:.0%} ({len(bearish)}), "
        f"Neutral: {len(neutral)/total:.0%} ({len(neutral)}) "
        f"| {sample_note}\n",
    ]

    if bullish:
        lines.append(f"Top {min(top_n, len(bullish))} Bullish posts (ranked by upvotes × confidence):")
        for p in bullish[:top_n]:
            lines.append(
                f"  [{p['source']}] [conf:{p['finbert_conf']}] [score:{p['score']}] "
                f"{p['text'][:150]}"
            )

    if bearish:
        lines.append(f"\nTop {min(top_n, len(bearish))} Bearish posts (ranked by upvotes × confidence):")
        for p in bearish[:top_n]:
            lines.append(
                f"  [{p['source']}] [conf:{p['finbert_conf']}] [score:{p['score']}] "
                f"{p['text'][:150]}"
            )

    return "\n".join(lines)
