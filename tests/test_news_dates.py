"""Regression tests for news date handling (refactor_plan Phase 2)."""

import pytest

from tradingagents.dataflows.alpha_vantage_common import format_datetime_for_api
from tradingagents.dataflows.backtest_cache import _filter_av_feed_by_date


def test_av_start_date_formats_to_midnight():
    assert format_datetime_for_api("2024-01-15") == "20240115T0000"


def test_av_end_date_includes_whole_day():
    """End-of-range timestamps must cover the full day, not exclude it at T0000."""
    assert format_datetime_for_api("2024-01-15", is_end=True) == "20240115T2359"


def test_av_feed_filter_keeps_end_date_articles():
    raw = {"items": "4", "feed": [
        {"title": "early", "time_published": "20240114T093000"},
        {"title": "on-start", "time_published": "20240115T000100"},
        {"title": "on-end-evening", "time_published": "20240116T220000"},
        {"title": "after", "time_published": "20240117T000100"},
    ]}
    out = _filter_av_feed_by_date(raw, "2024-01-15", "2024-01-16")
    titles = [a["title"] for a in out["feed"]]
    assert titles == ["on-start", "on-end-evening"]
    assert out["items"] == "2"


def test_av_feed_filter_drops_missing_timestamp():
    raw = {"items": "1", "feed": [{"title": "no-ts"}]}
    out = _filter_av_feed_by_date(raw, "2024-01-15", "2024-01-16")
    assert out["feed"] == []
