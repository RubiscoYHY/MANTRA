"""Regression tests for social data date handling and subject prefiltering
(refactor_plan Phase 3)."""

from datetime import datetime, timedelta, timezone

from tradingagents.agents.utils.social_data_tools import (
    _filter_recent_posts,
    subject_prefilter,
)


def _post(text, days_ago, ref=None, title=None):
    ref = ref or datetime.now(timezone.utc)
    ts = (ref - timedelta(days=days_ago)).timestamp()
    p = {"text": text, "timestamp": ts}
    if title is not None:
        p["title"] = title
    return p


class TestTradeDateRecencyFilter:
    def test_filters_relative_to_trade_date_not_now(self):
        """In backtest, recency must be measured against trade_date (no look-ahead)."""
        trade_date = "2026-01-15"
        ref = datetime(2026, 1, 15, tzinfo=timezone.utc)
        posts = [
            _post("inside window", days_ago=1, ref=ref),
            _post("too old", days_ago=10, ref=ref),
            # Published AFTER the trade date — future data, must be excluded.
            _post("from the future", days_ago=-2, ref=ref),
        ]
        out = _filter_recent_posts(posts, trade_date=trade_date)
        texts = [p["text"] for p in out]
        assert texts == ["inside window"]

    def test_default_uses_today(self):
        posts = [_post("fresh", days_ago=0), _post("stale", days_ago=30)]
        out = _filter_recent_posts(posts)
        assert [p["text"] for p in out] == ["fresh"]


class TestSubjectPrefilter:
    def test_cashtag_match_passes(self):
        posts = [{"text": "Loving $AAPL today", "title": ""}]
        assert subject_prefilter(posts, "AAPL") == posts

    def test_ticker_in_title_passes(self):
        posts = [{"text": "thoughts?", "title": "AAPL earnings preview"}]
        assert subject_prefilter(posts, "AAPL") == posts

    def test_dominant_ticker_passes(self):
        posts = [{"text": "AAPL AAPL AAPL is better than MSFT", "title": ""}]
        assert subject_prefilter(posts, "AAPL") == posts

    def test_passing_mention_of_other_subject_dropped(self):
        posts = [{"text": "MSFT MSFT MSFT will crush it, also AAPL exists", "title": ""}]
        assert subject_prefilter(posts, "AAPL") == []

    def test_no_mention_dropped(self):
        posts = [{"text": "I like turtles", "title": ""}]
        assert subject_prefilter(posts, "AAPL") == []
