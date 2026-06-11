"""Tests for GUI helpers and route registration (gui/app.py).

These cover the report-markdown assembly, backtest-eval serialization, the
agent day-rollover content check, and the presence of the new API routes.
They do not execute the analysis pipeline.
"""

import gui.app as g


def test_routes_registered():
    rules = {str(r.rule) for r in g.app.url_map.iter_rules()}
    for expected in (
        "/api/agent_view/<job_id>/<agent>",
        "/api/backtest_metrics/<job_id>",
        "/api/backtest_asset/<job_id>/<asset>",
        "/api/backtest_eval/<job_id>",
        "/api/backtest_save/<job_id>",
        "/api/run_record/<job_id>",
        "/api/run_days/<ticker>",
    ):
        assert expected in rules, f"missing route {expected}"


def test_build_report_md():
    md = g._build_report_md(
        {"signal": "BUY", "confidence": 0.8, "horizon": "1-5d"},
        "Go long.",
        {"market_report": "Market looks good.", "news_report": "Positive news."},
    )
    assert "# Signal: BUY" in md
    assert "Confidence: 80%" in md
    assert "Horizon: 1-5d" in md
    assert "## Decision" in md and "Go long." in md
    assert "## Market Analysis" in md and "Market looks good." in md
    assert "## News Analysis" in md and "Positive news." in md


def test_build_report_md_minimal():
    md = g._build_report_md(None, "", {})
    assert isinstance(md, str)
    assert md.endswith("\n")


def test_serialize_backtest_eval_empty():
    assert g._serialize_backtest_eval({"n_days": 0}) is None
    assert g._serialize_backtest_eval({}) is None


def test_serialize_backtest_eval():
    ev = {
        "n_days": 3,
        "metrics": {
            "strategy": {"total_return": 0.12, "annualized_return": 0.5,
                         "max_drawdown": -0.05, "sharpe": 1.2, "sortino": 1.8,
                         "volatility_annualized": 0.2},
            "buy_hold": {"total_return": 0.08, "annualized_return": 0.3,
                         "max_drawdown": -0.1, "sharpe": 0.9, "sortino": 1.1,
                         "volatility_annualized": 0.25},
        },
        "accuracy": 0.6,
        "turnover": 2.5,
    }
    out = g._serialize_backtest_eval(ev)
    assert out["n_days"] == 3
    labels = [r["metric"] for r in out["rows"]]
    for expected in ("Total return", "Annualized return", "Max drawdown",
                     "Sharpe", "Sortino", "Volatility (ann.)",
                     "Directional accuracy", "Turnover (sum |Δpos|)"):
        assert expected in labels
    # Directional accuracy formatted as a percentage; buy&hold column is em dash.
    acc_row = next(r for r in out["rows"] if r["metric"] == "Directional accuracy")
    assert acc_row["strategy"] == "60.0%"
    assert acc_row["buy_hold"] == "—"


def test_agent_has_content():
    assert g._agent_has_content({"status": "completed"}) is True
    assert g._agent_has_content({"status": "error"}) is True
    assert g._agent_has_content({"status": "pending"}) is False
    assert g._agent_has_content({"status": "in_progress"}) is False
    assert g._agent_has_content(
        {"status": "in_progress", "llm_turns": [{"x": 1}]}) is True
    assert g._agent_has_content(None) is False
    assert g._agent_has_content("not a dict") is False


def test_agent_view_unknown_job():
    client = g.app.test_client()
    resp = client.get("/api/agent_view/nope/Market%20Analyst")
    assert resp.status_code == 404


def test_backtest_asset_unknown_job():
    client = g.app.test_client()
    resp = client.get("/api/backtest_asset/nope/equity_curve")
    assert resp.status_code == 404


# --- Interactive (deferred-save) backtest completion flow -------------------

def test_parse_eval_params_defaults_and_clamps():
    # Missing / invalid → defaults.
    d = g._parse_eval_params({})
    assert d == {"overweight": 0.75, "underweight": 0.25,
                 "cost_bps": 5.0, "conf_gate": 0.55}
    d2 = g._parse_eval_params({"overweight": "nonsense"})
    assert d2["overweight"] == 0.75
    # Out-of-range → clamped to the documented bounds.
    d3 = g._parse_eval_params(
        {"overweight": "5", "underweight": "-1",
         "cost_bps": "1000", "conf_gate": "2"})
    assert d3 == {"overweight": 1.0, "underweight": 0.0,
                  "cost_bps": 100.0, "conf_gate": 1.0}


def test_eval_to_payload_shape():
    rows = [
        {"ticker": "AAPL", "date": "2024-01-02", "signal": "BUY",
         "confidence": 0.9, "actual_return": 0.01},
        {"ticker": "AAPL", "date": "2024-01-03", "signal": "SELL",
         "confidence": 0.9, "actual_return": -0.01},
        {"ticker": "AAPL", "date": "2024-01-04", "signal": "BUY",
         "confidence": 0.9, "actual_return": 0.02},
    ]
    from tradingagents.graph.backtest_eval import evaluate_backtest
    payload = g._eval_to_payload(evaluate_backtest(rows))
    assert payload["n_days"] == 3
    assert isinstance(payload["table"], list) and payload["table"]
    eq = payload["equity"]
    assert len(eq["dates"]) == 3
    assert len(eq["strategy"]) == 3 and len(eq["buy_hold"]) == 3
    assert all(isinstance(x, float) for x in eq["strategy"])


def _seed_backtest_job(job_id="jobx"):
    """Register a finished backtest job with retained per-ticker rows."""
    g._jobs[job_id] = {
        "status": "done",
        "backtest_rows": {
            "AAPL": [
                {"ticker": "AAPL", "date": "2024-01-02", "signal": "BUY",
                 "confidence": 0.9, "actual_return": 0.01},
                {"ticker": "AAPL", "date": "2024-01-03", "signal": "SELL",
                 "confidence": 0.9, "actual_return": -0.01},
            ],
            "MSFT": [
                {"ticker": "MSFT", "date": "2024-01-02", "signal": "HOLD",
                 "confidence": 0.9, "actual_return": 0.005},
                {"ticker": "MSFT", "date": "2024-01-03", "signal": "BUY",
                 "confidence": 0.9, "actual_return": 0.02},
            ],
        },
        "backtest_start": "2024-01-02",
        "backtest_end": "2024-01-04",
    }
    return job_id


def test_backtest_eval_route():
    job_id = _seed_backtest_job("jobeval")
    client = g.app.test_client()
    resp = client.get(
        f"/api/backtest_eval/{job_id}"
        "?ticker=AAPL&overweight=0.8&underweight=0.2&cost_bps=10&conf_gate=0.6")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ticker"] == "AAPL"
    assert data["n_days"] == 2
    assert len(data["equity"]["dates"]) == 2
    assert data["params"]["overweight"] == 0.8
    g._jobs.pop(job_id, None)


def test_backtest_eval_unknown_ticker_404():
    job_id = _seed_backtest_job("jobeval2")
    client = g.app.test_client()
    resp = client.get(f"/api/backtest_eval/{job_id}?ticker=ZZZZ")
    assert resp.status_code == 404
    g._jobs.pop(job_id, None)


def test_backtest_eval_unknown_job_404():
    client = g.app.test_client()
    resp = client.get("/api/backtest_eval/nope?ticker=AAPL")
    assert resp.status_code == 404


def test_backtest_save_route(tmp_path, monkeypatch):
    # Redirect the results dir so the test does not pollute the repo.
    monkeypatch.setattr(g, "RESULTS_DIR", str(tmp_path))
    job_id = _seed_backtest_job("jobsave")
    client = g.app.test_client()
    resp = client.post(
        f"/api/backtest_save/{job_id}",
        json={"overweight": 0.75, "underweight": 0.25,
              "cost_bps": 5, "conf_gate": 0.55})
    assert resp.status_code == 200
    saved = resp.get_json()["saved"]
    # Every ticker gets its own results folder; assets land on disk now (not
    # at completion time).
    assert set(saved.keys()) == {"AAPL", "MSFT"}
    for ticker, out_dir in saved.items():
        from pathlib import Path
        assert (Path(out_dir) / "signals.csv").exists()
        assert (Path(out_dir) / "metrics.csv").exists()
    g._jobs.pop(job_id, None)


def test_backtest_save_unknown_job_404():
    client = g.app.test_client()
    resp = client.post("/api/backtest_save/nope", json={})
    assert resp.status_code == 404
