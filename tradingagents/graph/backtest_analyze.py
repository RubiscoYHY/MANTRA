"""
backtest_analyze.py — TradingAgents backtest performance visualisation.

Replicates the evaluation methodology of the TradingAgents paper (2412.20138):
  - Metrics : Cumulative Return, Annualised Return, Sharpe Ratio, Max Drawdown
  - Baselines: Buy & Hold, MACD(12/26/9), KDJ+RSI, ZMR, SMA(5/20)

Position logic (bounded-stack state machine):
  Each BUY / OVERWEIGHT increments state by +1, capped at +1 (fully long).
  Each SELL / UNDERWEIGHT decrements state by −1, capped at −1 (fully short).
  HOLD leaves state unchanged.
  Transitions require two signals to cross from extreme to extreme:
    +1 → SELL → 0 → SELL → −1  (exit long first, then go short)
    −1 → BUY  → 0 → BUY  → +1  (exit short first, then go long)

Three TradingAgents variants:
  TA-Signal   : bounded-stack state machine driven by signal alone
  TA-Filtered : same, but only executes when confidence >= threshold
  TA-Scaled   : confidence-weighted state machine (our extension; position ∝ confidence)

Output (default — no --output flag):
  results/{TICKER}-{start_date}-{end_date}/analysis.png
  results/{TICKER}-{start_date}-{end_date}/metrics.csv

Usage
-----
# After running a CLI backtest:
    python -m tradingagents.graph.backtest_analyze results/backtest_AAPL_*.csv

# Quick demo (synthetic data, no API needed):
    python -m tradingagents.graph.backtest_analyze --demo

# Save figure to a specific path instead of auto-folder:
    python -m tradingagents.graph.backtest_analyze results/backtest_AAPL_*.csv --output reports/AAPL.png

Options
-------
--demo           Use synthetic 60-day data (no CSV / API needed).
--ticker TICKER  Override ticker symbol used for price download.
--threshold N    Confidence threshold for TA-Filtered (default: 0.65).
--capital N      Starting capital in USD (default: 10 000).
--rf N           Annualised risk-free rate for Sharpe (default: 0.05 = 5 %).
--output FILE    Save figure to FILE (overrides auto-folder default).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

matplotlib.rcParams.update({
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.30,
    "font.size":         9,
})

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BULLISH    = {"BUY", "OVERWEIGHT"}
_BEARISH    = {"SELL", "UNDERWEIGHT"}
_DIRECTIONAL = _BULLISH | _BEARISH

# Colour / style per strategy
_STYLES: dict[str, dict] = {
    "Buy & Hold":     {"color": "#666666", "lw": 1.4, "ls": "--",  "zorder": 2},
    "SMA (5/20)":     {"color": "#E67E22", "lw": 1.4, "ls": "-.",  "zorder": 2},
    "MACD":           {"color": "#8E44AD", "lw": 1.4, "ls": ":",   "zorder": 2},
    "KDJ+RSI":        {"color": "#16A085", "lw": 1.4, "ls": (0,(3,1,1,1)), "zorder": 2},
    "ZMR":            {"color": "#7F8C8D", "lw": 1.4, "ls": (0,(5,2)), "zorder": 2},
    "TA-Signal":      {"color": "#2980B9", "lw": 2.0, "ls": "-",   "zorder": 4},
    "TA-Filtered":    {"color": "#27AE60", "lw": 2.0, "ls": "-",   "zorder": 4},
    "TA-Scaled":      {"color": "#C0392B", "lw": 2.5, "ls": "-",   "zorder": 5},
}


# ---------------------------------------------------------------------------
# Price / OHLCV data
# ---------------------------------------------------------------------------

def _download_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download OHLCV from yfinance. Returns DataFrame with columns:
    Open, High, Low, Close, Volume (adjusted, sort ascending).
    Raises ValueError when no data is returned.
    """
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No price data found for {ticker} ({start} → {end}).")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    return raw.sort_index()


def _align_series(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """
    Reindex series onto the requested DatetimeIndex, forward-filling gaps
    (weekends / holidays that may appear in the target index).
    """
    combined = series.reindex(series.index.union(index)).ffill()
    return combined.reindex(index)


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

def _compute_kdj(high: pd.Series, low: pd.Series, close: pd.Series,
                 n: int = 9, m: int = 3) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    KDJ (Stochastic Oscillator).
    RSV = (Close - Low_n) / (High_n - Low_n) * 100
    K   = (m-1)/m * K_prev + 1/m * RSV
    D   = (m-1)/m * D_prev + 1/m * K
    J   = 3K - 2D
    Initial K and D both start at 50.
    """
    low_n  = low.rolling(n).min()
    high_n = high.rolling(n).max()
    rsv    = ((close - low_n) / (high_n - low_n + 1e-9) * 100).fillna(50.0)

    k_arr = np.full(len(rsv), 50.0)
    d_arr = np.full(len(rsv), 50.0)
    w     = 1.0 / m
    for i in range(1, len(rsv)):
        k_arr[i] = (1 - w) * k_arr[i - 1] + w * rsv.iloc[i]
        d_arr[i] = (1 - w) * d_arr[i - 1] + w * k_arr[i]

    k = pd.Series(k_arr, index=close.index)
    d = pd.Series(d_arr, index=close.index)
    j = 3 * k - 2 * d
    return k, d, j


def _compute_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI using exponential smoothing."""
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=n - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=n - 1, adjust=False).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)


# ---------------------------------------------------------------------------
# Baseline position arrays
# (all long-only: position ∈ {0.0, 1.0} each day)
# ---------------------------------------------------------------------------

def _positions_bah(n: int) -> np.ndarray:
    """Buy & Hold: fully invested every day."""
    return np.ones(n)


def _positions_sma(close: pd.Series, fast: int = 5, slow: int = 20) -> np.ndarray:
    """
    SMA crossover — long-short, matching paper Table 1.

    fast SMA > slow SMA → position = +1 (long)
    fast SMA < slow SMA → position = −1 (short)

    Uses previous day's values to eliminate look-ahead bias.
    Warm-up period before slow SMA is available: position = 0 (flat).
    """
    sma_f   = close.rolling(fast).mean().shift(1)
    sma_s   = close.rolling(slow).mean().shift(1)
    pos     = np.where(sma_f > sma_s, 1.0, np.where(sma_f < sma_s, -1.0, 0.0))
    # NaN rows (warm-up) → flat
    nan_mask = sma_f.isna() | sma_s.isna()
    pos[nan_mask.values] = 0.0
    return pos


def _positions_macd(close: pd.Series,
                    fast: int = 12, slow: int = 26, sig: int = 9) -> np.ndarray:
    """
    MACD crossover — long-short, matching paper Table 1.

    MACD line > signal line → position = +1 (long)
    MACD line < signal line → position = −1 (short)

    Uses previous day's values to eliminate look-ahead bias.
    Warm-up period (< slow periods): position = 0 (flat).
    """
    ema_f    = close.ewm(span=fast, adjust=False).mean()
    ema_s    = close.ewm(span=slow, adjust=False).mean()
    macd     = ema_f - ema_s
    sig_line = macd.ewm(span=sig, adjust=False).mean()
    diff     = macd.shift(1) - sig_line.shift(1)
    pos      = np.where(diff > 0, 1.0, np.where(diff < 0, -1.0, 0.0))
    # Warm-up: first (slow-1) rows have unreliable EMA → flat
    pos[:slow - 1] = 0.0
    return pos


def _positions_kdj_rsi(high: pd.Series, low: pd.Series, close: pd.Series) -> np.ndarray:
    """
    KDJ + RSI combined — long-short state machine, matching paper Table 1.

    BUY  trigger (+1): K golden-crosses D in oversold zone (K < 30)  OR  RSI < 30.
    SELL trigger (−1): K death-crosses  D in overbought zone (K > 70) OR  RSI > 70.
    Otherwise: maintain last position (0 until first signal).

    All indicator values are shifted by 1 day to eliminate look-ahead bias.
    """
    k, d, _j = _compute_kdj(high, low, close)
    rsi       = _compute_rsi(close)

    k1, d1 = k.shift(1), d.shift(1)   # yesterday's K, D
    k2, d2 = k.shift(2), d.shift(2)   # two days ago

    rsi1 = rsi.shift(1)

    buy_signal  = ((k1 > d1) & (k2 <= d2) & (k1 < 30)) | (rsi1 < 30)
    sell_signal = ((k1 < d1) & (k2 >= d2) & (k1 > 70)) | (rsi1 > 70)

    state = 0   # start flat; transitions: +1 (long) or −1 (short)
    pos   = []
    for i in range(len(close)):
        if buy_signal.iloc[i]:
            state = 1
        elif sell_signal.iloc[i]:
            state = -1
        # neither: maintain
        pos.append(float(state))
    return np.array(pos)


def _positions_zmr(close: pd.Series, window: int = 20,
                   threshold: float = 1.0) -> np.ndarray:
    """
    Zero Mean Reversion — long-short state machine, matching paper Table 1.

    Measures deviation of log-returns from their rolling mean (≈ 0):
        z = (log_ret − rolling_mean) / rolling_std

    z (prev day) < −threshold → price dropped sharply → expect bounce → LONG  (+1)
    z (prev day) > +threshold → price rose   sharply → expect pullback → SHORT (−1)
    Otherwise: maintain last position (0 until first signal).

    Uses previous-day z-score to eliminate look-ahead bias.
    """
    log_ret  = np.log(close / close.shift(1))
    roll_mu  = log_ret.rolling(window).mean()
    roll_sig = log_ret.rolling(window).std()
    z        = (log_ret - roll_mu) / (roll_sig + 1e-9)
    z1       = z.shift(1)   # previous day

    buy_signal  = z1 < -threshold
    sell_signal = z1 >  threshold

    state = 0   # start flat; transitions: +1 (long) or −1 (short)
    pos   = []
    for i in range(len(close)):
        if buy_signal.iloc[i]:
            state = 1
        elif sell_signal.iloc[i]:
            state = -1
        # neither: maintain
        pos.append(float(state))
    return np.array(pos)


# ---------------------------------------------------------------------------
# TradingAgents — bounded-stack state machine
# ---------------------------------------------------------------------------

def _positions_ta_bounded_stack(signals: pd.Series) -> np.ndarray:
    """
    Bounded-stack state machine for TA-Signal.

    Each BUY  increments state by +1, capped at +1.
    Each SELL decrements state by −1, capped at −1.
    HOLD leaves state unchanged.

    This means crossing from +1 to −1 requires two SELL signals
    (first exits long to flat, then goes short), and vice versa.
    Starts flat (0) until the first directional signal.
    """
    state = 0
    pos   = []
    for sig in signals:
        if sig in _BULLISH:
            state = min(state + 1, 1)
        elif sig in _BEARISH:
            state = max(state - 1, -1)
        pos.append(float(state))
    return np.array(pos)


def _positions_ta_bounded_stack_filtered(
    signals: pd.Series,
    confidences: pd.Series,
    threshold: float,
) -> np.ndarray:
    """
    Bounded-stack state machine for TA-Filtered.
    Same as _positions_ta_bounded_stack but transitions only execute
    when confidence >= threshold; sub-threshold signals behave as HOLD.
    """
    state = 0
    pos   = []
    for sig, conf in zip(signals, confidences):
        if float(conf) >= threshold:
            if sig in _BULLISH:
                state = min(state + 1, 1)
            elif sig in _BEARISH:
                state = max(state - 1, -1)
        pos.append(float(state))
    return np.array(pos)


def _positions_ta_scaled(signals: pd.Series, confidences: pd.Series) -> np.ndarray:
    """
    TA-Scaled: long-short state machine with confidence-weighted position size.

        weight = clip((confidence − 0.50) × 2, 0, 1)

        state = +1 (long)  → position = +weight   ∈ (0, +1]
        state = −1 (short) → position = −weight   ∈ [−1, 0)
        state =  0 (flat)  → position =  0

    Direction (state) is set by BUY/SELL/HOLD as in TA-Signal.
    Confidence only controls magnitude, not direction.

    Examples:
        BUY,  conf=0.50 → weight=0.00, position= 0.00  (no trade)
        BUY,  conf=0.75 → weight=0.50, position=+0.50  (half long)
        BUY,  conf=1.00 → weight=1.00, position=+1.00  (full long)
        SELL, conf=0.80 → weight=0.60, position=−0.60  (60% short)
    """
    state = 0
    pos   = []
    for sig, conf in zip(signals, confidences):
        if sig in _BULLISH:
            state = 1
        elif sig in _BEARISH:
            state = -1
        weight = float(np.clip((conf - 0.50) * 2.0, 0.0, 1.0))
        pos.append(state * weight)
    return np.array(pos)


# ---------------------------------------------------------------------------
# Portfolio simulation
# ---------------------------------------------------------------------------

def _simulate(positions: np.ndarray, daily_returns: np.ndarray,
              initial_capital: float = 10_000.0) -> np.ndarray:
    """
    Simulate portfolio value given position fractions and daily returns.

    positions[t]     : fraction of capital invested at end-of-day t
    daily_returns[t] : (price[t+1] - price[t]) / price[t]
                       (the return earned by holding positions[t] overnight)

    Returns value time-series of length len(positions) + 1,
    where index 0 = initial_capital (before any trade).
    """
    assert len(positions) == len(daily_returns), \
        f"Length mismatch: positions={len(positions)}, returns={len(daily_returns)}"
    growth     = 1.0 + positions * daily_returns
    cum_growth = np.concatenate([[1.0], np.cumprod(growth)])
    return initial_capital * cum_growth


# ---------------------------------------------------------------------------
# Performance metrics  (paper Appendix S1.2)
# ---------------------------------------------------------------------------

def _metrics(values: np.ndarray, rf_annual: float = 0.05) -> dict:
    """
    Compute the four metrics from the paper (S1.2).

    CR  = (V_end / V_start − 1) × 100 %
    AR  = ((V_end / V_start) ^ (1/N) − 1) × 100 %  where N = years
    SR  = (mean_excess_daily / std_daily) × sqrt(252)
    MDD = max((Peak − Trough) / Peak) × 100 %
    """
    n_days = len(values) - 1
    if n_days <= 0:
        return {"CR": 0.0, "AR": 0.0, "SR": 0.0, "MDD": 0.0}

    ret_daily  = np.diff(values) / values[:-1]
    cr         = (values[-1] / values[0] - 1.0) * 100.0
    n_yr       = n_days / 252.0
    ar         = ((values[-1] / values[0]) ** (1.0 / max(n_yr, 1e-9)) - 1.0) * 100.0
    rf_daily   = (1.0 + rf_annual) ** (1.0 / 252.0) - 1.0
    excess     = ret_daily - rf_daily
    sr         = excess.mean() / (excess.std(ddof=1) + 1e-9) * np.sqrt(252)
    peak       = np.maximum.accumulate(values)
    drawdown   = (values - peak) / peak
    mdd        = abs(drawdown.min()) * 100.0
    return {"CR": cr, "AR": ar, "SR": sr, "MDD": mdd}


# ---------------------------------------------------------------------------
# Calibration statistics
# ---------------------------------------------------------------------------

def _calibration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute directional accuracy per confidence bucket.
    Only directional signals (BUY/OW/SELL/UW) are included; HOLD/ERROR excluded.
    """
    sub = df[df["signal"].isin(_DIRECTIONAL)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["bucket", "midpoint", "n", "accuracy"])

    sub["correct"] = (
        (sub["signal"].isin(_BULLISH) & (sub["actual_return"] > 0))
        | (sub["signal"].isin(_BEARISH) & (sub["actual_return"] < 0))
    ).astype(int)

    bins   = [0.50, 0.60, 0.70, 0.80, 0.90, 1.01]
    labels = ["0.50–0.60", "0.60–0.70", "0.70–0.80", "0.80–0.90", "0.90–1.00"]
    sub["bucket"] = pd.cut(sub["confidence"], bins=bins, labels=labels, right=False)

    grp = (
        sub.groupby("bucket", observed=False)["correct"]
        .agg(n="count", accuracy="mean")
        .reset_index()
    )
    mid_map = {lb: lo + 0.05 for lb, lo in
               zip(labels, [0.50, 0.60, 0.70, 0.80, 0.90])}
    grp["midpoint"] = grp["bucket"].map(mid_map)
    return grp


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------

def _demo_data(n: int = 60, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic backtest results for --demo mode.
    Returns a DataFrame matching the CLI backtest CSV schema.
    """
    rng = np.random.default_rng(seed)
    _SIGS = ["BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL"]
    _PROB = [0.28, 0.18, 0.28, 0.14, 0.12]
    signals     = rng.choice(_SIGS, p=_PROB, size=n)
    confidences = np.clip(rng.normal(0.70, 0.13, n), 0.51, 0.99)

    # Slight edge: high-conf directional signals are more often correct
    base_ret = rng.normal(5e-4, 0.014, n)
    dates    = pd.bdate_range("2024-01-02", periods=n)
    return pd.DataFrame({
        "ticker":        "DEMO",
        "date":          [d.strftime("%Y-%m-%d") for d in dates],
        "signal":        signals,
        "confidence":    confidences.round(3),
        "horizon":       "1-5d",
        "actual_return": base_ret,
    })


# ---------------------------------------------------------------------------
# Main visualisation
# ---------------------------------------------------------------------------

def _build_figure(df: pd.DataFrame, ticker: str, ohlcv: pd.DataFrame | None,
                  initial_capital: float, threshold: float,
                  rf_annual: float) -> tuple[plt.Figure, dict]:
    """Construct and return (figure, values_dict) where values_dict maps strategy name → portfolio array."""

    # ── 0. Prepare data ────────────────────────────────────────────────────
    df = df.copy()
    df["date"]          = pd.to_datetime(df["date"])
    df["actual_return"] = pd.to_numeric(df["actual_return"], errors="coerce").fillna(0.0)
    df["confidence"]    = pd.to_numeric(df["confidence"],    errors="coerce").fillna(0.70)
    df                  = df.sort_values("date").reset_index(drop=True)

    n    = len(df)
    ret  = df["actual_return"].values   # shared return array for all strategies

    # ── 1. Price data for baselines ────────────────────────────────────────
    if ohlcv is not None and len(ohlcv) >= 26:
        trade_idx = pd.DatetimeIndex(df["date"])
        close     = _align_series(ohlcv["Close"], trade_idx)
        high      = _align_series(ohlcv["High"],  trade_idx)
        low       = _align_series(ohlcv["Low"],   trade_idx)
        has_ohlcv = True
    else:
        close = high = low = None
        has_ohlcv = False
        print("[WARN] No OHLCV data — baseline strategies unavailable (only B&H).")

    # ── 2. TradingAgents positions (bounded-stack state machine) ──────────
    pos_ta_sig  = _positions_ta_bounded_stack(df["signal"])
    pos_ta_filt = _positions_ta_bounded_stack_filtered(
        df["signal"], df["confidence"], threshold
    )
    pos_ta_sca  = _positions_ta_scaled(df["signal"], df["confidence"])

    # ── 3. Baseline positions ──────────────────────────────────────────────
    if has_ohlcv:
        pos_bah  = _positions_bah(n)
        pos_sma  = _positions_sma(close)
        pos_macd = _positions_macd(close)
        pos_kdj  = _positions_kdj_rsi(high, low, close)
        pos_zmr  = _positions_zmr(close)
    else:
        pos_bah  = _positions_bah(n)
        pos_sma  = pos_macd = pos_kdj = pos_zmr = None

    # ── 4. Portfolio values ────────────────────────────────────────────────
    def _val(pos):
        return _simulate(pos, ret, initial_capital) if pos is not None else None

    values = {
        "Buy & Hold":  _val(pos_bah),
        "SMA (5/20)":  _val(pos_sma),
        "MACD":        _val(pos_macd),
        "KDJ+RSI":     _val(pos_kdj),
        "ZMR":         _val(pos_zmr),
        "TA-Signal":   _val(pos_ta_sig),
        "TA-Filtered": _val(pos_ta_filt),
        "TA-Scaled":   _val(pos_ta_sca),
    }

    # Date axis: one entry before first trade (initial state) + one per trade date
    port_dates = (
        [df["date"].iloc[0] - pd.Timedelta(days=1)] + df["date"].tolist()
    )

    # ── 4. Metrics table (stdout) ─────────────────────────────────────────
    _print_metrics(values, ticker, threshold, rf_annual, initial_capital)

    # ── 5. Figure layout ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 11), constrained_layout=False)
    fig.suptitle(
        f"TradingAgents vs Traditional Strategies  ·  {ticker}  "
        f"({df['date'].iloc[0].strftime('%Y-%m-%d')} → "
        f"{df['date'].iloc[-1].strftime('%Y-%m-%d')})",
        fontsize=12, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.52, wspace=0.35,
        top=0.93, bottom=0.07, left=0.07, right=0.97,
    )
    ax_eq  = fig.add_subplot(gs[0:2, :])   # top 2 rows: full-width equity curves
    ax_pos = fig.add_subplot(gs[2, 0:2])   # bottom-left 2/3: TA position bar
    ax_cal = fig.add_subplot(gs[2, 2])     # bottom-right 1/3: calibration

    # ── Panel A: Equity curves ────────────────────────────────────────────
    def _pct(v: np.ndarray) -> np.ndarray:
        return (v / initial_capital - 1.0) * 100.0

    handles = []
    for name, v in values.items():
        if v is None:
            continue
        st   = _STYLES[name]
        line, = ax_eq.plot(
            port_dates, _pct(v),
            color=st["color"], lw=st["lw"], ls=st["ls"], zorder=st["zorder"],
            label=name,
        )
        # Right-side final return annotation
        ax_eq.annotate(
            f"{_pct(v)[-1]:+.1f}%",
            xy=(port_dates[-1], _pct(v)[-1]),
            xytext=(5, 0), textcoords="offset points",
            color=st["color"], fontsize=7.5, va="center",
        )
        handles.append(line)

    ax_eq.axhline(0, color="black", lw=0.8, alpha=0.45, zorder=1)
    ax_eq.set_ylabel("Cumulative Return (%)", fontsize=9)
    ax_eq.set_title(
        "Strategy Comparison — Cumulative Returns  "
        "(TA-Signal/Filtered: bounded-stack  |  TA-Scaled: confidence-weighted)",
        fontsize=9,
    )
    ax_eq.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax_eq.tick_params(axis="x", rotation=15)
    ax_eq.legend(handles=handles, loc="upper left", fontsize=8, ncol=4,
                 framealpha=0.85)

    # Shade TA-Scaled vs B&H gap
    v_bah = values.get("Buy & Hold")
    v_sca = values.get("TA-Scaled")
    if v_bah is not None and v_sca is not None:
        p_bah = _pct(v_bah)
        p_sca = _pct(v_sca)
        ax_eq.fill_between(
            port_dates, p_bah, p_sca,
            where=(p_sca > p_bah), alpha=0.10, color="#C0392B", label="_",
        )
        ax_eq.fill_between(
            port_dates, p_bah, p_sca,
            where=(p_sca < p_bah), alpha=0.10, color="#2980B9", label="_",
        )

    # ── Panel B: TA-Signal daily position (bounded-stack) ─────────────────
    sig_colors = [
        "#C0392B" if p > 0 else "#2980B9" if p < 0 else "#AAAAAA"
        for p in pos_ta_sig
    ]
    ax_pos.bar(df["date"], pos_ta_sig, color=sig_colors, width=0.8, alpha=0.8)
    ax_pos.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax_pos.set_ylabel("Position")
    ax_pos.set_title(
        "TA-Signal — Daily Position  (+1=Long, 0=Flat, −1=Short)  [bounded-stack]",
        fontsize=9,
    )
    ax_pos.set_ylim(-1.35, 1.35)
    ax_pos.tick_params(axis="x", rotation=15)

    # Overlay TA-Scaled as a step line (shows confidence-weighted magnitude)
    ax_pos.step(df["date"], pos_ta_sca, where="post",
                color="#27AE60", lw=1.5, alpha=0.85)
    ax_pos.legend(
        handles=[
            mpatches.Patch(color="#C0392B", label="Long  +1  (TA-Signal)"),
            mpatches.Patch(color="#2980B9", label="Short −1  (TA-Signal)"),
            mpatches.Patch(color="#AAAAAA", label="Flat   0  (TA-Signal)"),
            plt.Line2D([0], [0], color="#27AE60", lw=1.5,
                       label="TA-Scaled (conf-weighted)"),
        ],
        fontsize=7.5, loc="upper right",
    )

    # ── Panel C: Calibration curve ────────────────────────────────────────
    cal = _calibration(df)
    has_data = cal["n"].sum() > 0 if not cal.empty else False

    if has_data:
        x = np.arange(len(cal))
        colors_cal = [
            ("#27AE60" if (not np.isnan(row["accuracy"]) and
                          row["accuracy"] >= row["midpoint"] - 0.05)
             else "#E74C3C")
            for _, row in cal.iterrows()
        ]
        ax_cal.bar(x, cal["accuracy"].fillna(0), color=colors_cal, alpha=0.80, width=0.6)
        ax_cal.plot(x, cal["midpoint"], "k--o", lw=1.4, ms=4,
                    label="Ideal calibration")
        ax_cal.set_xticks(x)
        ax_cal.set_xticklabels(cal["bucket"], rotation=35, ha="right", fontsize=7)
        ax_cal.set_ylim(0, 1.05)
        ax_cal.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax_cal.set_title("Confidence Calibration\n(directional signals)", fontsize=9)
        ax_cal.legend(fontsize=7)

        for xi, (_, row) in zip(x, cal.iterrows()):
            if row["n"] > 0:
                ax_cal.text(xi, 0.03, f"n={row['n']}", ha="center",
                            va="bottom", fontsize=6.5, color="white", fontweight="bold")

        n_dir = int(cal["n"].sum())
        if n_dir < 50:
            ax_cal.set_xlabel(f"⚠ {n_dir} signals — low reliability",
                              fontsize=7.5, color="darkorange")
    else:
        ax_cal.text(0.5, 0.5, "No directional\nsignals recorded",
                    ha="center", va="center", transform=ax_cal.transAxes,
                    fontsize=10, color="gray")
        ax_cal.set_title("Confidence Calibration", fontsize=9)

    return fig, values


def _print_metrics(values: dict, ticker: str, threshold: float,
                   rf_annual: float, initial_capital: float) -> None:
    """Print a formatted metrics table (CR, AR, SR, MDD) to stdout."""
    SEP = "─" * 60
    print(f"\n{'═'*60}")
    print(f"  {ticker}  ·  Performance Summary")
    print(f"  TA-Signal/Filtered: bounded-stack state machine (±1, step-by-step transition)")
    print(f"  TA-Filtered threshold = {threshold:.2f}  |  "
          f"Rf = {rf_annual*100:.1f}%  |  "
          f"Initial capital = ${initial_capital:,.0f}")
    print(SEP)
    print(f"  {'Strategy':<18} {'CR %':>7} {'AR %':>7} {'SR':>6} {'MDD %':>7}")
    print(SEP)

    order = ["Buy & Hold", "SMA (5/20)", "MACD", "KDJ+RSI", "ZMR",
             "TA-Signal", "TA-Filtered", "TA-Scaled"]
    for name in order:
        v = values.get(name)
        if v is None:
            continue
        m = _metrics(v, rf_annual)
        print(f"  {name:<18} {m['CR']:>+6.2f}% {m['AR']:>+6.2f}% "
              f"{m['SR']:>5.2f}  {m['MDD']:>6.2f}%")
    print(SEP)
    print()


def _write_metrics_csv(values: dict, out_path: Path, rf_annual: float) -> None:
    """Save performance metrics to a CSV file."""
    import csv
    order = ["Buy & Hold", "SMA (5/20)", "MACD", "KDJ+RSI", "ZMR",
             "TA-Signal", "TA-Filtered", "TA-Scaled"]
    rows = []
    for name in order:
        v = values.get(name)
        if v is None:
            continue
        m = _metrics(v, rf_annual)
        rows.append({
            "strategy":  name,
            "CR_pct":    round(m["CR"],  4),
            "AR_pct":    round(m["AR"],  4),
            "SR":        round(m["SR"],  4),
            "MDD_pct":   round(m["MDD"], 4),
        })
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["strategy", "CR_pct", "AR_pct", "SR", "MDD_pct"])
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Argument parsing & entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TradingAgents backtest visualisation (paper methodology).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("csv",        nargs="?",           help="Backtest CSV from CLI.")
    p.add_argument("--demo",     action="store_true", help="Run with synthetic demo data.")
    p.add_argument("--ticker",   default=None,        help="Ticker symbol override.")
    p.add_argument("--threshold",type=float, default=0.65,
                   help="Confidence threshold for TA-Filtered (default 0.65).")
    p.add_argument("--capital",  type=float, default=10_000.0,
                   help="Initial capital USD (default 10 000).")
    p.add_argument("--rf",       type=float, default=0.05,
                   help="Annualised risk-free rate for Sharpe (default 0.05).")
    p.add_argument("--output",   default=None,
                   help="Save figure to this path (PNG/PDF/SVG).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Load backtest data ─────────────────────────────────────────────────
    if args.demo:
        df     = _demo_data()
        ticker = "DEMO"
    elif args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            sys.exit(f"[ERROR] CSV file not found: {csv_path}")
        df = pd.read_csv(csv_path)

        # Infer ticker
        if args.ticker:
            ticker = args.ticker.upper()
        elif "ticker" in df.columns:
            uniq = df["ticker"].unique().tolist()
            if len(uniq) > 1:
                print(f"[INFO] Multiple tickers: {uniq}. Analysing '{uniq[0]}'. "
                      "Use --ticker to override.")
                df = df[df["ticker"] == uniq[0]]
            ticker = str(df["ticker"].iloc[0]).upper()
        else:
            ticker = "UNKNOWN"
    else:
        sys.exit(
            "[ERROR] Provide a CSV path or use --demo.\n"
            "  python -m tradingagents.graph.backtest_analyze results/backtest_AAPL_*.csv\n"
            "  python -m tradingagents.graph.backtest_analyze --demo"
        )

    # ── Download OHLCV for baselines ───────────────────────────────────────
    ohlcv = None
    if ticker not in ("DEMO", "UNKNOWN") and not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        start  = df["date"].min().strftime("%Y-%m-%d")
        end_dt = df["date"].max() + pd.Timedelta(days=10)   # buffer
        end    = end_dt.strftime("%Y-%m-%d")
        try:
            print(f"[INFO] Downloading {ticker} OHLCV {start} → {end} …")
            ohlcv = _download_ohlcv(ticker, start, end)
            print(f"[INFO] {len(ohlcv)} trading days loaded.")
        except Exception as e:
            print(f"[WARN] Could not download OHLCV: {e}")
            print("[WARN] Baseline strategies will be unavailable.")

    # ── Build figure ───────────────────────────────────────────────────────
    fig, values = _build_figure(
        df              = df,
        ticker          = ticker,
        ohlcv           = ohlcv,
        initial_capital = args.capital,
        threshold       = args.threshold,
        rf_annual       = args.rf,
    )

    # ── Save figure + metrics CSV ──────────────────────────────────────────
    if args.output:
        # Explicit path override — save figure only
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"[INFO] Figure saved → {out.resolve()}")
        plt.close(fig)
    else:
        # Auto-create results/{TICKER}-{start_date}-{end_date}/
        df["date"] = pd.to_datetime(df["date"])
        start_str  = df["date"].min().strftime("%Y-%m-%d") if not df.empty else "unknown"
        end_str    = df["date"].max().strftime("%Y-%m-%d") if not df.empty else "unknown"
        out_dir    = Path("results") / f"{ticker}-{start_str}-{end_str}"
        out_dir.mkdir(parents=True, exist_ok=True)

        fig_path = out_dir / "analysis.png"
        fig.savefig(fig_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"[INFO] Figure saved → {fig_path.resolve()}")

        csv_path = out_dir / "metrics.csv"
        _write_metrics_csv(values, csv_path, args.rf)
        print(f"[INFO] Metrics CSV  → {csv_path.resolve()}")


if __name__ == "__main__":
    main()
