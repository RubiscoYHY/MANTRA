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

import time
import requests
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Annotated
from langchain_core.tools import tool

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
    cutoff = datetime.utcnow() - timedelta(days=_SOCIAL_RECENCY_DAYS)

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
                post_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).replace(tzinfo=None)
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


def _filter_recent_posts(posts: list[dict], max_age_days: int = _SOCIAL_RECENCY_DAYS) -> list[dict]:
    """
    Drop posts outside the recency window.
    Reddit timestamps are Unix epoch floats; StockTwits are ISO-8601 strings.
    Posts with missing or unparseable timestamps are kept (benefit of the doubt).
    """
    cutoff = datetime.utcnow() - timedelta(days=max_age_days)
    kept = []
    for p in posts:
        ts = p.get("timestamp", "")
        if not ts:
            kept.append(p)
            continue
        try:
            if isinstance(ts, (int, float)):
                post_dt = datetime.utcfromtimestamp(ts)
            else:
                post_dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).replace(tzinfo=None)
            if post_dt >= cutoff:
                kept.append(p)
        except (ValueError, OSError, OverflowError):
            kept.append(p)  # unparseable → keep
    return kept


def get_social_posts_cached(ticker: str) -> list[dict]:
    """
    Return combined StockTwits + Reddit posts for a ticker, filtered to the
    last _SOCIAL_RECENCY_DAYS days. Cached per ticker to avoid redundant
    fetches within one analysis run.
    """
    if ticker not in _posts_cache:
        raw = _fetch_stocktwits_raw(ticker) + _fetch_reddit_raw(ticker)
        _posts_cache[ticker] = _filter_recent_posts(raw)
    return _posts_cache[ticker]


def clear_posts_cache() -> None:
    """Clear the posts cache. Call between analysis runs for fresh data."""
    _posts_cache.clear()


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
) -> str:
    """
    Run FinBERT on a list of post dicts and return a compact structured summary.

    Input post dict keys:  text, source, score (upvotes), timestamp
    Output: ~600-token string with sentiment distribution + top posts per class.

    Falls back to a plain post listing if transformers is not installed.

    Args:
        posts: List of post dicts from _fetch_stocktwits_raw / _fetch_reddit_raw.
        min_neutral_confidence: Neutral posts below this confidence are dropped
                                as low-signal noise.
        top_n: Number of top posts to surface per sentiment class.
    """
    if not posts:
        return "No social media posts available for this ticker."

    _notify_finbert_status("in_progress")

    try:
        return _finbert_aggregate_inner(posts, min_neutral_confidence, top_n)
    finally:
        _notify_finbert_status("completed")


def _finbert_aggregate_inner(
    posts: list[dict],
    min_neutral_confidence: float,
    top_n: int,
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
    lines = [
        "=== Social Media Sentiment (FinBERT) ===",
        f"Distribution — Bullish: {len(bullish)/total:.0%} ({len(bullish)}), "
        f"Bearish: {len(bearish)/total:.0%} ({len(bearish)}), "
        f"Neutral: {len(neutral)/total:.0%} ({len(neutral)}) "
        f"| Total posts after filtering: {total}\n",
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
