"""Portfolio strategy evaluation for backtest signal series.

Strategy: "Confidence-Gated Rating Target" (long-only, daily).

    rating   → target exposure:  BUY 1.00 / OVERWEIGHT 0.75 /
                                  UNDERWEIGHT 0.25 / SELL 0.00
    HOLD (or confidence below the gate, or ERROR) → keep yesterday's position.

Why this shape for THIS project:
  * The pipeline's output is an advisory five-tier rating WITH a confidence
    and sizing semantics already built in ("overweight" literally means a
    partial position) — mapping ratings to target exposures uses all of that
    information directly instead of collapsing it back to +1/-1.
  * Long-only with a cash floor matches the intended user (daily-frequency,
    fundamentals-driven fund support): no borrow costs, no margin modeling,
    and a SELL means "step aside", which is how an advisory signal is
    actually consumed.
  * HOLD-keeps-position plus the confidence gate gives hysteresis: low-
    conviction flip-flops do not trade, which controls turnover — the main
    killer of daily signal strategies once costs are included.
  * Execution timing matches the data: the signal for day T is produced from
    information available on T, and `actual_return` in the backtest rows is
    the T→T+1 close return, so position(T) × actual_return(T) is causally
    clean by construction.

Costs are charged on turnover (default 5 bps per unit traded). Benchmark is
buy & hold of the same ticker over the same days.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252

# Default rating → target exposure map. BUY/SELL are the full/flat anchors;
# the intermediate OVERWEIGHT/UNDERWEIGHT weights are user-tunable (the GUI
# exposes them as interactive parameters).
DEFAULT_OVERWEIGHT = 0.75
DEFAULT_UNDERWEIGHT = 0.25


def _rating_targets(overweight: float, underweight: float) -> dict:
    return {
        "BUY": 1.00,
        "OVERWEIGHT": overweight,
        "HOLD": None,          # None = keep previous position
        "UNDERWEIGHT": underweight,
        "SELL": 0.00,
    }


def build_positions(
    results: list[dict],
    conf_gate: float = 0.55,
    initial_position: float = 0.0,
    overweight: float = DEFAULT_OVERWEIGHT,
    underweight: float = DEFAULT_UNDERWEIGHT,
) -> list[float]:
    """Daily exposure series from signal rows (see module docstring)."""
    targets = _rating_targets(overweight, underweight)
    positions: list[float] = []
    prev = initial_position
    for row in results:
        rating = str(row.get("signal", "HOLD")).upper()
        try:
            conf = float(row.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        target = targets.get(rating)
        if target is None or conf < conf_gate:
            cur = prev
        else:
            cur = target
        positions.append(cur)
        prev = cur
    return positions


def _annualized_return(equity: pd.Series) -> float:
    n = len(equity)
    if n == 0 or equity.iloc[-1] <= 0:
        return float("nan")
    return float(equity.iloc[-1] ** (TRADING_DAYS / n) - 1)


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1).min())


def _sharpe(daily: pd.Series, rf_annual: float) -> float:
    if daily.std(ddof=0) == 0:
        return float("nan")
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    return float((daily.mean() - rf_daily) / daily.std(ddof=0) * math.sqrt(TRADING_DAYS))


def _sortino(daily: pd.Series, rf_annual: float) -> float:
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    downside = daily[daily < rf_daily] - rf_daily
    if len(downside) == 0:
        return float("nan")
    dd = math.sqrt(float((downside ** 2).mean()))
    if dd == 0:
        return float("nan")
    return float((daily.mean() - rf_daily) / dd * math.sqrt(TRADING_DAYS))


def _metrics(daily: pd.Series, rf_annual: float) -> dict:
    equity = (1 + daily).cumprod()
    return {
        "total_return": float(equity.iloc[-1] - 1) if len(equity) else float("nan"),
        "annualized_return": _annualized_return(equity),
        "max_drawdown": _max_drawdown(equity),
        "sharpe": _sharpe(daily, rf_annual),
        "sortino": _sortino(daily, rf_annual),
        "volatility_annualized": float(daily.std(ddof=0) * math.sqrt(TRADING_DAYS)),
    }


def directional_accuracy(results: list[dict]) -> Optional[float]:
    """Share of directional calls (non-HOLD/ERROR) that matched the next-day
    return sign. BUY/OVERWEIGHT predict up; SELL/UNDERWEIGHT predict down."""
    correct = total = 0
    for row in results:
        rating = str(row.get("signal", "")).upper()
        try:
            r = float(row.get("actual_return") or 0.0)
        except (TypeError, ValueError):
            continue
        if rating in ("BUY", "OVERWEIGHT"):
            pred = 1
        elif rating in ("SELL", "UNDERWEIGHT"):
            pred = -1
        else:
            continue
        total += 1
        if (r > 0 and pred == 1) or (r < 0 and pred == -1):
            correct += 1
    return correct / total if total else None


def evaluate_backtest(
    results: list[dict],
    conf_gate: float = 0.55,
    cost_bps: float = 5.0,
    rf_annual: float = 0.05,
    overweight: float = DEFAULT_OVERWEIGHT,
    underweight: float = DEFAULT_UNDERWEIGHT,
) -> dict:
    """Evaluate the strategy vs buy & hold over backtest signal rows.

    Args:
        results: rows with keys ticker/date/signal/confidence/actual_return
                 (the list run_backtest() returns).

    Returns:
        {"equity": DataFrame[date, strategy, buy_hold, position],
         "metrics": {"strategy": {...}, "buy_hold": {...}},
         "accuracy": float|None, "turnover": float, "n_days": int}
    """
    rows = [r for r in results if r.get("date")]
    if not rows:
        return {"equity": pd.DataFrame(), "metrics": {}, "accuracy": None,
                "turnover": 0.0, "n_days": 0}

    returns = pd.Series(
        [float(r.get("actual_return") or 0.0) for r in rows], dtype=float
    )
    positions = pd.Series(
        build_positions(rows, conf_gate=conf_gate,
                        overweight=overweight, underweight=underweight),
        dtype=float,
    )
    trades = positions.diff().abs()
    trades.iloc[0] = abs(positions.iloc[0])
    cost = cost_bps / 10_000.0

    strat_daily = positions * returns - trades * cost
    bh_daily = returns.copy()

    equity = pd.DataFrame({
        "date": [r["date"] for r in rows],
        "strategy": (1 + strat_daily).cumprod(),
        "buy_hold": (1 + bh_daily).cumprod(),
        "position": positions,
    })

    return {
        "equity": equity,
        "metrics": {
            "strategy": _metrics(strat_daily, rf_annual),
            "buy_hold": _metrics(bh_daily, rf_annual),
        },
        "accuracy": directional_accuracy(rows),
        "turnover": float(trades.sum()),
        "n_days": len(rows),
    }


def save_backtest_assets(
    results: list[dict],
    out_dir: str,
    ticker: str,
    start_date: str,
    end_date: str,
    conf_gate: float = 0.55,
    cost_bps: float = 5.0,
    rf_annual: float = 0.05,
    overweight: float = DEFAULT_OVERWEIGHT,
    underweight: float = DEFAULT_UNDERWEIGHT,
) -> dict:
    """Evaluate and persist all backtest artifacts into out_dir.

    Writes equity_curve.png, metrics.csv, metrics.md, signals.csv.
    Returns {"evaluation": <evaluate_backtest dict>, "paths": {...}} so a UI
    can render the numbers without re-reading the files.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ev = evaluate_backtest(results, conf_gate=conf_gate,
                           cost_bps=cost_bps, rf_annual=rf_annual,
                           overweight=overweight, underweight=underweight)
    paths: dict[str, str] = {}

    pd.DataFrame(results).to_csv(out / "signals.csv", index=False)
    paths["signals"] = str(out / "signals.csv")

    if ev["n_days"]:
        m = ev["metrics"]
        rows = []
        labels = [
            ("total_return", "Total return", "{:+.2%}"),
            ("annualized_return", "Annualized return", "{:+.2%}"),
            ("max_drawdown", "Max drawdown", "{:.2%}"),
            ("sharpe", "Sharpe", "{:.2f}"),
            ("sortino", "Sortino", "{:.2f}"),
            ("volatility_annualized", "Volatility (ann.)", "{:.2%}"),
        ]
        for key, label, fmt in labels:
            rows.append({
                "metric": label,
                "strategy": fmt.format(m["strategy"][key]),
                "buy_hold": fmt.format(m["buy_hold"][key]),
            })
        acc = ev["accuracy"]
        rows.append({"metric": "Directional accuracy",
                     "strategy": f"{acc:.1%}" if acc is not None else "n/a",
                     "buy_hold": "—"})
        rows.append({"metric": "Turnover (sum |Δpos|)",
                     "strategy": f"{ev['turnover']:.2f}", "buy_hold": "—"})
        mdf = pd.DataFrame(rows)
        mdf.to_csv(out / "metrics.csv", index=False)
        paths["metrics_csv"] = str(out / "metrics.csv")
        md = [f"# Backtest {ticker} {start_date} → {end_date}", "",
              "| Metric | Strategy | Buy & Hold |",
              "|---|---|---|"]
        md += [f"| {r['metric']} | {r['strategy']} | {r['buy_hold']} |" for r in rows]
        (out / "metrics.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        paths["metrics_md"] = str(out / "metrics.md")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            eq = ev["equity"]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(eq["date"], eq["strategy"], label="Strategy (confidence-gated rating target)")
            ax.plot(eq["date"], eq["buy_hold"], label="Buy & Hold", linestyle="--")
            ax.set_title(f"{ticker} backtest {start_date} → {end_date}")
            ax.set_ylabel("Growth of $1")
            step = max(1, len(eq) // 10)
            ax.set_xticks(range(0, len(eq), step))
            ax.tick_params(axis="x", rotation=45)
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out / "equity_curve.png", dpi=120)
            plt.close(fig)
            paths["equity_curve"] = str(out / "equity_curve.png")
        except Exception:
            logger.warning("equity curve rendering failed", exc_info=True)

    return {"evaluation": ev, "paths": paths}
