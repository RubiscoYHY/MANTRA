from .alpha_vantage_common import _make_api_request, format_datetime_for_api

def get_news(ticker, start_date, end_date) -> dict[str, str] | str:
    """Returns live and historical market news & sentiment data from premier news outlets worldwide.

    Covers stocks, cryptocurrencies, forex, and topics like fiscal policy, mergers & acquisitions, IPOs.

    Args:
        ticker: Stock symbol for news articles.
        start_date: Start date for news search.
        end_date: End date for news search.

    Returns:
        Dictionary containing news sentiment data or JSON string.
    """
    from .backtest_cache import get_backtest_cache
    _cache = get_backtest_cache()
    if _cache.is_active():
        cached = _cache.get_av_news(ticker, start_date, end_date)
        if cached is not None:
            return cached
    params = {
        "tickers": ticker,
        "time_from": format_datetime_for_api(start_date),
        "time_to": format_datetime_for_api(end_date),
    }

    return _make_api_request("NEWS_SENTIMENT", params)

def get_global_news(curr_date, look_back_days: int = 7, limit: int = 50) -> dict[str, str] | str:
    """Returns global market news & sentiment data without ticker-specific filtering.

    Covers broad market topics like financial markets, economy, and more.

    Args:
        curr_date: Current date in yyyy-mm-dd format.
        look_back_days: Number of days to look back (default 7).
        limit: Maximum number of articles (default 50).

    Returns:
        Dictionary containing global news sentiment data or JSON string.
    """
    from .backtest_cache import get_backtest_cache
    _cache = get_backtest_cache()
    if _cache.is_active():
        cached = _cache.get_av_global_news(curr_date, look_back_days)
        if cached is not None:
            return cached
    from datetime import datetime, timedelta

    # Calculate start date
    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = curr_dt - timedelta(days=look_back_days)
    start_date = start_dt.strftime("%Y-%m-%d")

    params = {
        "topics": "financial_markets,economy_macro,economy_monetary",
        "time_from": format_datetime_for_api(start_date),
        "time_to": format_datetime_for_api(curr_date),
        "limit": str(limit),
    }

    return _make_api_request("NEWS_SENTIMENT", params)


def get_insider_transactions(
    symbol: str,
    curr_date: str = None,
) -> dict[str, str] | str:
    """Returns historical insider transactions filtered to on-or-before curr_date.

    Covers transactions by founders, executives, board members, etc.

    Args:
        symbol: Ticker symbol. Example: "IBM".
        curr_date: Current trading date YYYY-MM-DD; transactions after this date
                   are excluded to prevent look-ahead bias.

    Returns:
        Dictionary containing insider transaction data or JSON string.
    """
    from .backtest_cache import get_backtest_cache
    _cache = get_backtest_cache()
    if _cache.is_active():
        cached = _cache.get_av_insider_transactions(symbol, curr_date)
        if cached is not None:
            return cached
    raw = _make_api_request("INSIDER_TRANSACTIONS", {"symbol": symbol})
    if curr_date:
        from .backtest_cache import _filter_av_insider_by_date
        raw = _filter_av_insider_by_date(raw, curr_date)
    return raw