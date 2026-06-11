"""Tests for the confidence-gated rating-target strategy evaluation."""

import tempfile
from pathlib import Path

from tradingagents.graph.backtest_eval import (
    build_positions,
    directional_accuracy,
    evaluate_backtest,
    save_backtest_assets,
)


def _row(date, signal, conf, ret):
    return {"ticker": "TEST", "date": date, "signal": signal,
            "confidence": conf, "horizon": "1-5d", "actual_return": ret}


ROWS = [
    _row("2026-01-02", "BUY", 0.80, 0.02),         # enter full
    _row("2026-01-05", "HOLD", 0.90, 0.01),        # keep 1.0
    _row("2026-01-06", "SELL", 0.40, -0.03),       # below gate -> keep 1.0
    _row("2026-01-07", "UNDERWEIGHT", 0.70, -0.01),  # cut to 0.25
    _row("2026-01-08", "SELL", 0.75, -0.02),       # exit to 0.0
    _row("2026-01-09", "ERROR", 0.0, 0.04),        # carry 0.0
]


def test_position_logic():
    assert build_positions(ROWS) == [1.0, 1.0, 1.0, 0.25, 0.0, 0.0]


def test_tunable_rating_weights():
    rows = [_row("d1", "OVERWEIGHT", 0.9, 0.0), _row("d2", "UNDERWEIGHT", 0.9, 0.0)]
    assert build_positions(rows, overweight=0.6, underweight=0.1) == [0.6, 0.1]
    ev = evaluate_backtest(ROWS, overweight=1.0, underweight=0.0)
    # With OVERWEIGHT=1.0/UNDERWEIGHT=0.0 the strategy becomes binary.
    assert set(ev["equity"]["position"].unique()) <= {0.0, 1.0}


def test_hold_and_low_confidence_do_not_trade():
    rows = [_row("d1", "BUY", 0.9, 0.0), _row("d2", "SELL", 0.54, 0.0),
            _row("d3", "HOLD", 0.99, 0.0)]
    assert build_positions(rows, conf_gate=0.55) == [1.0, 1.0, 1.0]


def test_directional_accuracy_excludes_hold():
    # BUY:+2% correct, SELL(low conf still counts as a directional call? no —
    # accuracy is about the RATING itself): SELL on -3% correct,
    # UNDERWEIGHT on -1% correct, SELL on -2% correct, HOLD/ERROR excluded.
    acc = directional_accuracy(ROWS)
    assert acc == 1.0


def test_evaluation_metrics_shape():
    ev = evaluate_backtest(ROWS)
    assert ev["n_days"] == 6
    for side in ("strategy", "buy_hold"):
        m = ev["metrics"][side]
        for key in ("total_return", "annualized_return", "max_drawdown",
                    "sharpe", "sortino"):
            assert key in m
    # Strategy went flat before the last +4% day, so it must lag buy & hold
    # on that day but have a shallower max drawdown overall.
    assert ev["metrics"]["strategy"]["max_drawdown"] >= ev["metrics"]["buy_hold"]["max_drawdown"]
    assert len(ev["equity"]) == 6


def test_costs_reduce_returns():
    free = evaluate_backtest(ROWS, cost_bps=0.0)
    costly = evaluate_backtest(ROWS, cost_bps=50.0)
    assert costly["metrics"]["strategy"]["total_return"] < \
           free["metrics"]["strategy"]["total_return"]


def test_save_assets_writes_files():
    out = tempfile.mkdtemp()
    res = save_backtest_assets(ROWS, out, "TEST", "2026-01-02", "2026-01-09")
    assert Path(res["paths"]["metrics_csv"]).exists()
    assert Path(res["paths"]["metrics_md"]).exists()
    assert Path(res["paths"]["signals"]).exists()
    assert Path(res["paths"]["equity_curve"]).exists()
    assert res["evaluation"]["accuracy"] == 1.0


def test_empty_results():
    ev = evaluate_backtest([])
    assert ev["n_days"] == 0 and ev["metrics"] == {}
