"""Backtest data cache — pre-fetch all required data once per ticker.

During backtesting each trading day re-invokes the full agent pipeline,
which would normally trigger dozens of identical API calls.  This module
fetches everything upfront (once per ticker), then serves date-filtered
results from memory so no future data leaks to the agents.

Usage (from cli/main.py):
    from tradingagents.dataflows.backtest_cache import get_backtest_cache
    cache = get_backtest_cache()
    cache.initialize(ticker, start_date, end_date)   # before inner loop
    ...
    cache.clear()                                    # after inner loop
"""

import copy
import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class BacktestDataCache:
    """Pre-fetch and cache all market data for one ticker's backtest period.

    Data is fetched once, stored in memory, and returned with a
    date-filter applied on each query so agents never see future data.
    """

    def __init__(self) -> None:
        self._active: bool = False
        self._ticker: str = ""
        self._start_date: str = ""
        self._end_date: str = ""
        self._last_trading_day: str = ""  # last actual trading day ≤ end_date
        self._store: dict = {}

    # ------------------------------------------------------------------ #
    # Life-cycle                                                           #
    # ------------------------------------------------------------------ #

    def is_active(self) -> bool:
        return self._active

    def initialize(self, ticker: str, start_date: str, end_date: str) -> None:
        """Pre-fetch all data needed for one ticker's backtest run."""
        self._active = True
        self._ticker = ticker.upper()
        self._start_date = start_date
        self._end_date = end_date
        self._store = {}

        self._prefetch_next_day_returns()
        self._prefetch_yf_financials()
        self._prefetch_yf_news()
        self._prefetch_yf_global_news()
        self._prefetch_av_financials()
        self._prefetch_av_news()
        self._prefetch_av_global_news()

    def clear(self) -> None:
        """Release all cached data after a ticker's backtest completes."""
        self._active = False
        self._store.clear()

    # ------------------------------------------------------------------ #
    # Pre-fetch helpers                                                    #
    # ------------------------------------------------------------------ #

    def _prefetch_next_day_returns(self) -> None:
        """Download the full price series once and pre-compute daily returns.

        Also determines _last_trading_day — the last actual trading day that
        falls within [start_date, end_date] — used for the fundamentals gate.
        """
        try:
            end_plus = (
                pd.Timestamp(self._end_date) + pd.tseries.offsets.BDay(5)
            ).strftime("%Y-%m-%d")
            hist = yf.download(
                self._ticker,
                start=self._start_date,
                end=end_plus,
                auto_adjust=True,
                progress=False,
                multi_level_index=False,
            )
            if len(hist) < 2:
                self._store["next_day_returns"] = {}
                self._last_trading_day = self._end_date
                return

            closes = hist["Close"]

            # Identify the last trading day within [start_date, end_date]
            in_range = [
                closes.index[i]
                for i in range(len(closes))
                if closes.index[i].strftime("%Y-%m-%d") <= self._end_date
            ]
            self._last_trading_day = (
                in_range[-1].strftime("%Y-%m-%d") if in_range else self._end_date
            )

            returns: dict[str, float] = {}
            for i in range(len(closes) - 1):
                date_str = closes.index[i].strftime("%Y-%m-%d")
                prev = float(closes.iloc[i])
                nxt = float(closes.iloc[i + 1])
                returns[date_str] = (nxt - prev) / prev if prev != 0 else 0.0
            self._store["next_day_returns"] = returns
        except Exception as exc:
            logger.warning("Backtest cache: failed to prefetch returns: %s", exc)
            self._store["next_day_returns"] = {}
            self._last_trading_day = self._end_date

    def _prefetch_yf_financials(self) -> None:
        """Fetch yfinance fundamentals and financial statements once."""
        try:
            from .stockstats_utils import yf_retry

            t = yf.Ticker(self._ticker)
            self._store["yf_info"] = yf_retry(lambda: t.info) or {}
            self._store["yf_balance_q"] = yf_retry(lambda: t.quarterly_balance_sheet)
            self._store["yf_balance_a"] = yf_retry(lambda: t.balance_sheet)
            self._store["yf_cashflow_q"] = yf_retry(lambda: t.quarterly_cashflow)
            self._store["yf_cashflow_a"] = yf_retry(lambda: t.cashflow)
            self._store["yf_income_q"] = yf_retry(lambda: t.quarterly_income_stmt)
            self._store["yf_income_a"] = yf_retry(lambda: t.income_stmt)
            self._store["yf_insider"] = yf_retry(lambda: t.insider_transactions)
        except Exception as exc:
            logger.warning("Backtest cache: yf financials prefetch failed: %s", exc)

    def _prefetch_yf_news(self) -> None:
        """Fetch yfinance news articles once (up to 200)."""
        try:
            from .stockstats_utils import yf_retry

            t = yf.Ticker(self._ticker)
            self._store["yf_news"] = yf_retry(lambda: t.get_news(count=200)) or []
        except Exception as exc:
            logger.warning("Backtest cache: yf news prefetch failed: %s", exc)
            self._store["yf_news"] = []

    def _prefetch_yf_global_news(self) -> None:
        """Fetch yfinance global market news once."""
        try:
            from .stockstats_utils import yf_retry

            queries = [
                "stock market economy",
                "Federal Reserve interest rates",
                "inflation economic outlook",
                "global markets trading",
            ]
            all_articles: list = []
            seen: set = set()
            for q in queries:
                search = yf_retry(
                    lambda _q=q: yf.Search(
                        query=_q, news_count=50, enable_fuzzy_query=True
                    )
                )
                for article in getattr(search, "news", []):
                    content = article.get("content", {})
                    title = content.get("title", article.get("title", ""))
                    if title and title not in seen:
                        seen.add(title)
                        all_articles.append(article)
            self._store["yf_global_news"] = all_articles
        except Exception as exc:
            logger.warning("Backtest cache: yf global news prefetch failed: %s", exc)
            self._store["yf_global_news"] = []

    def _prefetch_av_financials(self) -> None:
        """Fetch Alpha Vantage fundamentals and statements once."""
        try:
            from .alpha_vantage_common import _make_api_request

            sym = self._ticker
            self._store["av_overview"] = _make_api_request("OVERVIEW", {"symbol": sym})
            self._store["av_balance"] = _make_api_request("BALANCE_SHEET", {"symbol": sym})
            self._store["av_cashflow"] = _make_api_request("CASH_FLOW", {"symbol": sym})
            self._store["av_income"] = _make_api_request("INCOME_STATEMENT", {"symbol": sym})
            self._store["av_insider"] = _make_api_request(
                "INSIDER_TRANSACTIONS", {"symbol": sym}
            )
        except Exception as exc:
            logger.warning("Backtest cache: AV financials prefetch failed: %s", exc)

    def _prefetch_av_news(self) -> None:
        """Fetch Alpha Vantage ticker news for the full backtest window."""
        try:
            from .alpha_vantage_common import _make_api_request, format_datetime_for_api

            params = {
                "tickers": self._ticker,
                "time_from": format_datetime_for_api(self._start_date),
                "time_to": format_datetime_for_api(self._end_date),
                "limit": "200",
            }
            self._store["av_news"] = _make_api_request("NEWS_SENTIMENT", params)
        except Exception as exc:
            logger.warning("Backtest cache: AV news prefetch failed: %s", exc)
            self._store["av_news"] = None

    def _prefetch_av_global_news(self) -> None:
        """Fetch Alpha Vantage global news for the full backtest window."""
        try:
            from .alpha_vantage_common import _make_api_request, format_datetime_for_api

            params = {
                "topics": "financial_markets,economy_macro,economy_monetary",
                "time_from": format_datetime_for_api(self._start_date),
                "time_to": format_datetime_for_api(self._end_date),
                "limit": "200",
            }
            self._store["av_global_news"] = _make_api_request("NEWS_SENTIMENT", params)
        except Exception as exc:
            logger.warning("Backtest cache: AV global news prefetch failed: %s", exc)
            self._store["av_global_news"] = None

    # ------------------------------------------------------------------ #
    # Query interface — each method applies the appropriate date-filter   #
    # ------------------------------------------------------------------ #

    def _ticker_matches(self, ticker: str) -> bool:
        return ticker.upper() == self._ticker

    def get_next_day_return(self, date: str) -> float:
        """Return the pre-computed next-day return for a given trade date."""
        return self._store.get("next_day_returns", {}).get(date, 0.0)

    # -- yfinance fundamentals ----------------------------------------- #

    def get_yf_fundamentals(self, ticker: str, curr_date: str = "") -> "str | None":
        """Return cached yfinance fundamentals, gated to the final trading day.

        Real-time metrics (P/E, market cap, etc.) reflect today's values, not
        the historical simulation date.  Releasing them on every backtest day
        would inject look-ahead bias.  They are withheld until curr_date reaches
        the last actual trading day in the backtest window (_last_trading_day).
        """
        if not self._ticker_matches(ticker):
            return None
        if curr_date and self._last_trading_day and curr_date < self._last_trading_day:
            return (
                f"[Backtest] Fundamentals overview withheld on {curr_date}. "
                f"Real-time metrics (P/E, market cap, etc.) reflect the data-fetch "
                f"date rather than the simulation date, so they are only injected on "
                f"the final trading day ({self._last_trading_day}) to avoid "
                f"look-ahead bias. Use get_balance_sheet / get_income_statement / "
                f"get_cashflow for period-accurate fundamental analysis."
            )
        info = self._store.get("yf_info")
        if not info:
            return None
        fields = [
            ("Name", info.get("longName")),
            ("Sector", info.get("sector")),
            ("Industry", info.get("industry")),
            ("Market Cap", info.get("marketCap")),
            ("PE Ratio (TTM)", info.get("trailingPE")),
            ("Forward PE", info.get("forwardPE")),
            ("PEG Ratio", info.get("pegRatio")),
            ("Price to Book", info.get("priceToBook")),
            ("EPS (TTM)", info.get("trailingEps")),
            ("Forward EPS", info.get("forwardEps")),
            ("Dividend Yield", info.get("dividendYield")),
            ("Beta", info.get("beta")),
            ("52 Week High", info.get("fiftyTwoWeekHigh")),
            ("52 Week Low", info.get("fiftyTwoWeekLow")),
            ("50 Day Average", info.get("fiftyDayAverage")),
            ("200 Day Average", info.get("twoHundredDayAverage")),
            ("Revenue (TTM)", info.get("totalRevenue")),
            ("Gross Profit", info.get("grossProfits")),
            ("EBITDA", info.get("ebitda")),
            ("Net Income", info.get("netIncomeToCommon")),
            ("Profit Margin", info.get("profitMargins")),
            ("Operating Margin", info.get("operatingMargins")),
            ("Return on Equity", info.get("returnOnEquity")),
            ("Return on Assets", info.get("returnOnAssets")),
            ("Debt to Equity", info.get("debtToEquity")),
            ("Current Ratio", info.get("currentRatio")),
            ("Book Value", info.get("bookValue")),
            ("Free Cash Flow", info.get("freeCashflow")),
        ]
        lines = [f"{lbl}: {val}" for lbl, val in fields if val is not None]
        header = (
            f"# Company Fundamentals for {ticker.upper()}\n"
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        return header + "\n".join(lines)

    def _get_yf_financial_df(
        self, store_key: str, ticker: str, curr_date: "str | None"
    ) -> "pd.DataFrame | None":
        """Return a date-filtered copy of a cached yfinance financial DataFrame."""
        if not self._ticker_matches(ticker):
            return None
        raw = self._store.get(store_key)
        if raw is None or (hasattr(raw, "empty") and raw.empty):
            return None
        from .stockstats_utils import filter_financials_by_date
        return filter_financials_by_date(raw, curr_date)

    def get_yf_balance_sheet(
        self, ticker: str, freq: str, curr_date: "str | None"
    ) -> "str | None":
        key = "yf_balance_q" if freq.lower() == "quarterly" else "yf_balance_a"
        df = self._get_yf_financial_df(key, ticker, curr_date)
        if df is None:
            return None
        if df.empty:
            return f"No balance sheet data found for symbol '{ticker}'"
        header = (
            f"# Balance Sheet data for {ticker.upper()} ({freq})\n"
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        return header + df.to_csv()

    def get_yf_cashflow(
        self, ticker: str, freq: str, curr_date: "str | None"
    ) -> "str | None":
        key = "yf_cashflow_q" if freq.lower() == "quarterly" else "yf_cashflow_a"
        df = self._get_yf_financial_df(key, ticker, curr_date)
        if df is None:
            return None
        if df.empty:
            return f"No cash flow data found for symbol '{ticker}'"
        header = (
            f"# Cash Flow data for {ticker.upper()} ({freq})\n"
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        return header + df.to_csv()

    def get_yf_income_statement(
        self, ticker: str, freq: str, curr_date: "str | None"
    ) -> "str | None":
        key = "yf_income_q" if freq.lower() == "quarterly" else "yf_income_a"
        df = self._get_yf_financial_df(key, ticker, curr_date)
        if df is None:
            return None
        if df.empty:
            return f"No income statement data found for symbol '{ticker}'"
        header = (
            f"# Income Statement data for {ticker.upper()} ({freq})\n"
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        return header + df.to_csv()

    def get_yf_insider_transactions(
        self, ticker: str, curr_date: "str | None" = None
    ) -> "str | None":
        """Return cached insider transactions filtered to on-or-before curr_date."""
        if not self._ticker_matches(ticker):
            return None
        data = self._store.get("yf_insider")
        if data is None:
            return None
        if hasattr(data, "empty") and data.empty:
            return f"No insider transactions data found for symbol '{ticker}'"
        filtered = _filter_insider_df_by_date(data, curr_date) if curr_date else data
        if hasattr(filtered, "empty") and filtered.empty:
            return f"No insider transactions found for symbol '{ticker}' on or before {curr_date}"
        header = (
            f"# Insider Transactions data for {ticker.upper()}\n"
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        return header + filtered.to_csv()

    # -- yfinance news -------------------------------------------------- #

    def get_yf_news(
        self, ticker: str, start_date: str, end_date: str
    ) -> "str | None":
        """Return cached yfinance news filtered to [start_date, end_date]."""
        if not self._ticker_matches(ticker):
            return None
        articles = self._store.get("yf_news")
        if articles is None:
            return None

        from .yfinance_news import _extract_article_data
        from dateutil.relativedelta import relativedelta

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        end_plus = end_dt + relativedelta(days=1)

        news_str = ""
        count = 0
        for article in articles:
            data = _extract_article_data(article)
            if data["pub_date"]:
                pub = data["pub_date"].replace(tzinfo=None)
                if not (start_dt <= pub <= end_plus):
                    continue
            news_str += f"### {data['title']} (source: {data['publisher']})\n"
            if data["summary"]:
                news_str += f"{data['summary']}\n"
            if data["link"]:
                news_str += f"Link: {data['link']}\n"
            news_str += "\n"
            count += 1

        if count == 0:
            return f"No news found for {ticker} between {start_date} and {end_date}"
        return f"## {ticker} News, from {start_date} to {end_date}:\n\n{news_str}"

    def get_yf_global_news(
        self, curr_date: str, look_back_days: int, limit: int
    ) -> "str | None":
        """Return cached yfinance global news filtered to the look-back window."""
        articles = self._store.get("yf_global_news")
        if articles is None:
            return None

        from .yfinance_news import _extract_article_data
        from dateutil.relativedelta import relativedelta

        curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        start_dt = curr_dt - relativedelta(days=look_back_days)
        start_date = start_dt.strftime("%Y-%m-%d")

        news_str = ""
        count = 0
        for article in articles:
            if "content" in article:
                data = _extract_article_data(article)
                pub = data.get("pub_date")
                if pub:
                    pub_naive = pub.replace(tzinfo=None) if hasattr(pub, "replace") else pub
                    # Strict future guard: skip anything after curr_date
                    if pub_naive > curr_dt + relativedelta(days=1):
                        continue
                    if pub_naive < start_dt:
                        continue
                title = data["title"]
                publisher = data["publisher"]
                link = data["link"]
                summary = data["summary"]
            else:
                title = article.get("title", "No title")
                publisher = article.get("publisher", "Unknown")
                link = article.get("link", "")
                summary = ""

            news_str += f"### {title} (source: {publisher})\n"
            if summary:
                news_str += f"{summary}\n"
            if link:
                news_str += f"Link: {link}\n"
            news_str += "\n"
            count += 1
            if count >= limit:
                break

        if not news_str:
            return f"No global news found for {curr_date}"
        return f"## Global Market News, from {start_date} to {curr_date}:\n\n{news_str}"

    # -- Alpha Vantage fundamentals ------------------------------------- #

    def get_av_fundamentals(self, ticker: str, curr_date: str = "") -> "object | None":
        """Return cached AV overview, gated to the final trading day (same policy as yf)."""
        if not self._ticker_matches(ticker):
            return None
        if curr_date and self._last_trading_day and curr_date < self._last_trading_day:
            return (
                f"[Backtest] Fundamentals overview withheld on {curr_date}. "
                f"Real-time metrics reflect the data-fetch date rather than the "
                f"simulation date, so they are only injected on the final trading day "
                f"({self._last_trading_day}) to avoid look-ahead bias. Use "
                f"get_balance_sheet / get_income_statement / get_cashflow instead."
            )
        return self._store.get("av_overview")

    def _get_av_financial_filtered(
        self, store_key: str, ticker: str, curr_date: "str | None"
    ) -> "dict | None":
        """Return a deep-copy of a cached AV financial dict with date filtering applied."""
        if not self._ticker_matches(ticker):
            return None
        raw = self._store.get(store_key)
        if raw is None:
            return None
        from .alpha_vantage_fundamentals import _filter_reports_by_date
        # Deep copy prevents mutating the cache on each daily call
        return _filter_reports_by_date(copy.deepcopy(raw), curr_date)

    def get_av_balance_sheet(
        self, ticker: str, curr_date: "str | None"
    ) -> "dict | None":
        return self._get_av_financial_filtered("av_balance", ticker, curr_date)

    def get_av_cashflow(
        self, ticker: str, curr_date: "str | None"
    ) -> "dict | None":
        return self._get_av_financial_filtered("av_cashflow", ticker, curr_date)

    def get_av_income_statement(
        self, ticker: str, curr_date: "str | None"
    ) -> "dict | None":
        return self._get_av_financial_filtered("av_income", ticker, curr_date)

    def get_av_insider_transactions(
        self, ticker: str, curr_date: "str | None" = None
    ) -> "object | None":
        """Return cached AV insider transactions filtered to on-or-before curr_date."""
        if not self._ticker_matches(ticker):
            return None
        raw = self._store.get("av_insider")
        if raw is None or not curr_date:
            return raw
        return _filter_av_insider_by_date(raw, curr_date)

    # -- Alpha Vantage news --------------------------------------------- #

    def get_av_news(
        self, ticker: str, start_date: str, end_date: str
    ) -> "dict | None":
        """Return cached AV ticker news filtered to [start_date, end_date]."""
        if not self._ticker_matches(ticker):
            return None
        raw = self._store.get("av_news")
        if raw is None:
            return None
        return _filter_av_feed_by_date(raw, start_date, end_date)

    def get_av_global_news(
        self, curr_date: str, look_back_days: int
    ) -> "dict | None":
        """Return cached AV global news filtered to the look-back window."""
        raw = self._store.get("av_global_news")
        if raw is None:
            return None
        start_dt = datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=look_back_days)
        return _filter_av_feed_by_date(raw, start_dt.strftime("%Y-%m-%d"), curr_date)


# --------------------------------------------------------------------------- #
# Module-level helpers                                                         #
# --------------------------------------------------------------------------- #

def _filter_av_feed_by_date(raw: dict, start_date: str, end_date: str) -> dict:
    """Return a new dict with the 'feed' list filtered to [start_date, end_date].

    AV timestamps are formatted as "20240115T130000".  We build matching
    boundary strings and compare lexicographically — no datetime parsing needed.
    """
    if not isinstance(raw, dict) or "feed" not in raw:
        return raw
    # Convert "2024-01-15" → "20240115T000000" / "20240115T235959"
    lo = start_date.replace("-", "") + "T000000"
    hi = end_date.replace("-", "") + "T235959"
    filtered = [a for a in raw["feed"] if lo <= a.get("time_published", "") <= hi]
    result = dict(raw)
    result["feed"] = filtered
    result["items"] = str(len(filtered))
    return result


def _filter_insider_df_by_date(data: pd.DataFrame, curr_date: str) -> pd.DataFrame:
    """Return rows of a yfinance insider-transactions DataFrame with date ≤ curr_date.

    yfinance uses "Start Date" as the transaction date column (the SEC filing
    effective date). Falls back to checking common alternate column names and
    finally the DataFrame index before returning unfiltered data.
    """
    cutoff = pd.Timestamp(curr_date)
    for col in ("Start Date", "Date", "startDate", "date", "Transaction Date"):
        if col in data.columns:
            dates = pd.to_datetime(data[col], errors="coerce")
            return data[dates <= cutoff]
    # Check if the index itself carries timestamps
    try:
        idx_dates = pd.to_datetime(data.index, errors="coerce")
        if idx_dates.notna().all():
            return data[idx_dates <= cutoff]
    except Exception:
        pass
    return data  # No recognisable date column — return unfiltered


def _filter_av_insider_by_date(raw: object, curr_date: str) -> object:
    """Filter Alpha Vantage INSIDER_TRANSACTIONS response to records ≤ curr_date.

    AV returns a dict with a "data" list.  Each record has a "transaction_date"
    field in ISO format (YYYY-MM-DDThh:mm:ss.sss or YYYY-MM-DD).  We compare
    the date prefix lexicographically against curr_date.
    """
    if not isinstance(raw, dict) or "data" not in raw:
        return raw
    filtered = [
        r for r in raw["data"]
        if str(r.get("transaction_date", ""))[:10] <= curr_date
    ]
    result = dict(raw)
    result["data"] = filtered
    return result


# Module-level singleton shared across all data-fetching modules
_instance = BacktestDataCache()


def get_backtest_cache() -> BacktestDataCache:
    """Return the process-wide BacktestDataCache singleton."""
    return _instance
