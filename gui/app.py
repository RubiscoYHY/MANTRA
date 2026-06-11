"""MANTRA GUI — Flask web interface for MANTRA trading analysis."""

from __future__ import annotations

import datetime
import json
import threading
import time
import uuid
import webbrowser
from collections import deque
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from flask import Flask, Response, jsonify, render_template, request

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients.model_catalog import MODEL_OPTIONS
from tradingagents.observability import load_record
from tradingagents.streaming import (
    ANALYST_ORDER,
    ANALYST_AGENT_NAMES,
    REPORT_SECTIONS,
    REPORT_SECTION_ORDER,
    REPORT_SECTION_TITLES,
    build_initial_agent_status,
)

# Runs directory for the on-disk run records (per ticker/date JSON files).
RUNS_DIR = DEFAULT_CONFIG.get("runs_dir", "./runs")
# Results directory for saved reports + backtest analytics assets.
RESULTS_DIR = DEFAULT_CONFIG.get("results_dir", "./results")


def _build_report_md(signal: dict | None, decision_text: str,
                     reports: dict, sections_order=None) -> str:
    """Assemble a human-readable markdown report for one analysis day.

    Layout: a signal/confidence/horizon header, the decision text, then each
    available report section under its title. ``reports`` maps section keys to
    text; ``signal`` is the processed signal dict (may be None).
    """
    sig = signal or {}
    lines: list[str] = []
    header_sig = sig.get("signal")
    if header_sig:
        lines.append(f"# Signal: {header_sig}")
        meta_bits = []
        conf = sig.get("confidence")
        if conf is not None:
            try:
                meta_bits.append(f"Confidence: {float(conf) * 100:.0f}%")
            except (TypeError, ValueError):
                pass
        if sig.get("horizon"):
            meta_bits.append(f"Horizon: {sig['horizon']}")
        if meta_bits:
            lines.append("  ·  ".join(meta_bits))
        lines.append("")
    if decision_text and str(decision_text).strip():
        lines.append("## Decision")
        lines.append(str(decision_text).strip())
        lines.append("")
    for key in (sections_order or REPORT_SECTION_ORDER):
        content = reports.get(key)
        if content and str(content).strip():
            title = REPORT_SECTION_TITLES.get(key, key)
            lines.append(f"## {title}")
            if isinstance(content, list):
                content = "\n".join(str(x) for x in content)
            lines.append(str(content).strip())
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _save_report_md(out_path: Path, signal: dict | None, decision_text: str,
                    reports: dict) -> None:
    """Write the assembled report markdown to ``out_path`` (best-effort)."""
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            _build_report_md(signal, decision_text, reports),
            encoding="utf-8",
        )
    except Exception:
        pass


app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)

# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------

_jobs: dict[str, dict[str, Any]] = {}


def _sse_event(event: str, data: Any) -> str:
    """Format a Server-Sent Event string."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


# ---------------------------------------------------------------------------
# Routes — pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------

@app.route("/api/models/<provider>/<mode>")
def get_models(provider: str, mode: str):
    """Return model options for a provider + mode (quick / deep)."""
    provider = provider.lower()
    if provider in MODEL_OPTIONS and mode in MODEL_OPTIONS[provider]:
        options = [
            {"label": label, "value": value}
            for label, value in MODEL_OPTIONS[provider][mode]
        ]
        return jsonify(options)
    return jsonify([])


@app.route("/api/providers")
def get_providers():
    """Return list of available LLM providers."""
    providers = [
        {"value": "openai", "label": "OpenAI"},
        {"value": "google", "label": "Google"},
        {"value": "anthropic", "label": "Anthropic"},
        {"value": "xai", "label": "xAI"},
        {"value": "openrouter", "label": "OpenRouter"},
        {"value": "huggingface", "label": "HuggingFace"},
        {"value": "ollama", "label": "Ollama"},
    ]
    return jsonify(providers)


@app.route("/api/start", methods=["POST"])
def start_analysis():
    """Start a new analysis job. Returns a job ID for SSE streaming."""
    payload = request.get_json(force=True)
    job_id = str(uuid.uuid4())[:8]
    q: Queue = Queue()

    _jobs[job_id] = {
        "queue": q,
        "status": "running",
        "thread": None,
        # Run-record bookkeeping (filled in once the job thread parses payload).
        "runs_dir": RUNS_DIR,
        "ticker": None,        # primary ticker under analysis
        "current_date": None,  # the trade date whose record we currently serve
    }

    t = threading.Thread(
        target=_run_analysis_job,
        args=(job_id, payload, q),
        daemon=True,
    )
    _jobs[job_id]["thread"] = t
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/api/stream/<job_id>")
def stream(job_id: str):
    """SSE endpoint — streams analysis events for a given job."""
    if job_id not in _jobs:
        return Response("Job not found", status=404)

    def generate():
        q = _jobs[job_id]["queue"]
        while True:
            try:
                event_type, data = q.get(timeout=30)
            except Empty:
                # Send heartbeat to keep connection alive
                yield ": heartbeat\n\n"
                continue

            yield _sse_event(event_type, data)

            if event_type in ("complete", "error"):
                break

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/run_record/<job_id>")
def run_record(job_id: str):
    """Return the CURRENT run record JSON for a job's ticker + trade date.

    The job tracks its ticker and (in backtest mode) the day currently being
    analyzed, so by default we serve that day's record. A ``?date=`` query
    param overrides the date. The file is re-read on every call — it is
    rewritten atomically by the recorder, so polling it is cheap and safe.
    Returns {} (with the keys present) when no record exists yet so the
    frontend can degrade gracefully.
    """
    job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404

    ticker = job.get("ticker")
    date = request.args.get("date") or job.get("current_date")
    runs_dir = job.get("runs_dir", RUNS_DIR)

    if not ticker or not date:
        return jsonify({"record": None, "ticker": ticker, "date": date,
                        "job_status": job.get("status")})

    record = load_record(runs_dir, ticker, date)
    return jsonify({
        "record": record,
        "ticker": ticker,
        "date": date,
        "job_status": job.get("status"),
    })


@app.route("/api/run_days/<ticker>")
def run_days(ticker: str):
    """List the trade dates that have a run record for this ticker.

    Powers the backtest date selector on the final-report page. Dates are the
    JSON filenames under runs/{TICKER}/, sorted ascending.
    """
    runs_dir = request.args.get("runs_dir") or RUNS_DIR
    ticker_dir = Path(runs_dir) / ticker.upper()
    days: list[str] = []
    if ticker_dir.is_dir():
        for p in ticker_dir.glob("*.json"):
            days.append(p.stem)
    days.sort()
    return jsonify({"ticker": ticker.upper(), "days": days})


def _agent_has_content(agent_data: Any) -> bool:
    """True if an agent record entry holds meaningful (non-pending) data.

    Used by the day-rollover fallback: a fresh day's record lists the watched
    agent as ``pending`` with empty sub-lists before it actually runs, which we
    do NOT want to display over the previous day's completed data.
    """
    if not isinstance(agent_data, dict):
        return False
    status = agent_data.get("status")
    if status in ("in_progress", "completed", "error"):
        # in_progress only counts once it has produced something.
        if status == "in_progress":
            return bool(
                agent_data.get("llm_turns")
                or agent_data.get("tool_calls")
                or agent_data.get("memory")
                or agent_data.get("output")
            )
        return True
    return False


@app.route("/api/agent_view/<job_id>/<agent>")
def agent_view(job_id: str, agent: str):
    """Resolve the most recent record carrying data for ``agent``.

    In backtest mode the served day advances; the freshly-rolled day lists the
    watched agent as pending/empty until it runs. To keep the panel stable we
    scan the current day first, then fall back to earlier days (descending)
    until we find one where the agent has real content. The response always
    states which date is being shown and whether it is the current served day.
    """
    job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404

    ticker = job.get("ticker")
    runs_dir = job.get("runs_dir", RUNS_DIR)
    current_date = job.get("current_date")

    if not ticker:
        return jsonify({"agent": agent, "date": None, "agent_data": None,
                        "is_current_day": True, "job_status": job.get("status")})

    # Build the descending list of candidate dates: current day first, then any
    # earlier dates present in the runs dir.
    ticker_dir = Path(runs_dir) / ticker.upper()
    all_days: list[str] = []
    if ticker_dir.is_dir():
        all_days = sorted(p.stem for p in ticker_dir.glob("*.json"))

    candidates: list[str] = []
    if current_date:
        candidates.append(current_date)
    # earlier days, newest first, excluding the current date
    for d in reversed(all_days):
        if d != current_date and (not current_date or d < current_date):
            candidates.append(d)

    for date in candidates:
        record = load_record(runs_dir, ticker, date)
        if not record:
            continue
        agent_data = (record.get("agents") or {}).get(agent)
        if _agent_has_content(agent_data):
            return jsonify({
                "agent": agent,
                "date": date,
                "agent_data": agent_data,
                "is_current_day": (date == current_date),
                "job_status": job.get("status"),
            })

    # Nothing with content yet — return the current day's (possibly pending) entry.
    record = load_record(runs_dir, ticker, current_date) if current_date else None
    agent_data = (record.get("agents") or {}).get(agent) if record else None
    return jsonify({
        "agent": agent,
        "date": current_date,
        "agent_data": agent_data,
        "is_current_day": True,
        "job_status": job.get("status"),
    })


@app.route("/api/backtest_metrics/<job_id>")
def backtest_metrics(job_id: str):
    """Return the backtest comparison table for a finished job.

    Exposes the available asset names (not server paths) so the client can
    request them via /api/backtest_asset/<job_id>/<asset>.
    """
    job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    bt = job.get("backtest_assets")
    if not bt:
        return jsonify({"available": False})
    return jsonify({
        "available": True,
        "ticker": bt.get("ticker"),
        "table": bt.get("table"),
        "assets": list((bt.get("paths") or {}).keys()),
    })


@app.route("/api/backtest_asset/<job_id>/<asset>")
def backtest_asset(job_id: str, asset: str):
    """Serve a backtest analytics file by logical name (no arbitrary paths).

    ``asset`` is one of the keys in the job's stored ``paths`` map (e.g.
    ``equity_curve``); the job → out_dir mapping is resolved server-side.
    """
    from flask import send_file

    job = _jobs.get(job_id)
    if job is None:
        return Response("Job not found", status=404)
    paths = (job.get("backtest_assets") or {}).get("paths") or {}
    file_path = paths.get(asset)
    if not file_path:
        return Response("Asset not found", status=404)
    p = Path(file_path)
    # Confine the served file to the job's recorded out_dir.
    out_dir = (job.get("backtest_assets") or {}).get("out_dir")
    try:
        resolved = p.resolve()
        if out_dir and Path(out_dir).resolve() not in resolved.parents:
            return Response("Forbidden", status=403)
    except Exception:
        pass
    if not p.exists():
        return Response("Asset file missing", status=404)
    return send_file(str(p))


def _parse_eval_params(source: dict) -> dict:
    """Parse + clamp the strategy parameters from a query/JSON mapping.

    overweight/underweight/conf_gate clamp to [0, 1]; cost_bps to [0, 100].
    Missing/invalid values fall back to the engine defaults.
    """
    def _f(key: str, default: float, lo: float, hi: float) -> float:
        raw = source.get(key)
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return default
        if v != v:  # NaN guard
            return default
        return max(lo, min(hi, v))

    return {
        "overweight": _f("overweight", 0.75, 0.0, 1.0),
        "underweight": _f("underweight", 0.25, 0.0, 1.0),
        "cost_bps": _f("cost_bps", 5.0, 0.0, 100.0),
        "conf_gate": _f("conf_gate", 0.55, 0.0, 1.0),
    }


@app.route("/api/backtest_eval/<job_id>")
def backtest_eval_route(job_id: str):
    """Recompute backtest analytics for one ticker with tunable parameters.

    Reads the per-ticker signal rows retained on the job and runs
    evaluate_backtest() with the supplied (validated) parameters. Nothing is
    written to disk — this powers the interactive completion view.
    """
    job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    rows_by_ticker = job.get("backtest_rows") or {}
    if not rows_by_ticker:
        return jsonify({"error": "No backtest data for job"}), 404

    ticker = (request.args.get("ticker") or "").upper()
    if not ticker:
        # Default to the first ticker when none is specified.
        ticker = next(iter(rows_by_ticker))
    rows = rows_by_ticker.get(ticker)
    if rows is None:
        return jsonify({"error": f"Unknown ticker {ticker}"}), 404

    params = _parse_eval_params(request.args)
    from tradingagents.graph.backtest_eval import evaluate_backtest

    ev = evaluate_backtest(rows, **params)
    payload = _eval_to_payload(ev)
    payload["ticker"] = ticker
    payload["params"] = params
    return jsonify(payload)


@app.route("/api/backtest_save/<job_id>", methods=["POST"])
def backtest_save_route(job_id: str):
    """Persist backtest analytics for EVERY ticker in the job.

    Body: {overweight, underweight, cost_bps, conf_gate}. For each ticker,
    save_backtest_assets() writes equity_curve.png / metrics.{csv,md} /
    signals.csv into results/{TICKER}_{start}_{end} using the chosen params.
    Returns {"saved": {ticker: out_dir}}.
    """
    job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    rows_by_ticker = job.get("backtest_rows") or {}
    if not rows_by_ticker:
        return jsonify({"error": "No backtest data for job"}), 404

    body = request.get_json(silent=True) or {}
    params = _parse_eval_params(body)
    start_date = job.get("backtest_start")
    end_date = job.get("backtest_end")

    from tradingagents.graph.backtest_eval import save_backtest_assets

    saved: dict[str, str] = {}
    errors: dict[str, str] = {}
    for ticker, rows in rows_by_ticker.items():
        out_dir = str(Path(RESULTS_DIR) / f"{ticker}_{start_date}_{end_date}")
        try:
            save_backtest_assets(
                rows, out_dir=out_dir, ticker=ticker,
                start_date=start_date, end_date=end_date, **params,
            )
            saved[ticker] = out_dir
        except Exception as e:  # noqa: BLE001
            errors[ticker] = str(e)
    resp = {"saved": saved}
    if errors:
        resp["errors"] = errors
    return jsonify(resp)


# ---------------------------------------------------------------------------
# Analysis runner (runs in background thread)
# ---------------------------------------------------------------------------

# Agent display name mapping, analyst ordering, and report-section metadata are
# imported from tradingagents.streaming (shared with the CLI).

# FIXED_AGENTS is GUI-specific: it surfaces a distinct "Judge" row that the CLI
# progress table does not track.
FIXED_AGENTS = {
    "Research Team": ["Bull Researcher", "Bear Researcher", "Judge", "Research Manager"],
    "Trading Team": ["Trader"],
    "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
    "Portfolio Management": ["Portfolio Manager"],
}


def _classify_message_type(message) -> tuple[str, str]:
    """Classify a LangGraph message into type and content."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    if isinstance(message, HumanMessage):
        return "Human", str(message.content)
    elif isinstance(message, ToolMessage):
        content = str(message.content)
        if len(content) > 200:
            content = content[:200] + "..."
        return "Tool", content
    elif isinstance(message, AIMessage):
        content = str(message.content) if message.content else ""
        return "AI", content
    else:
        return "System", str(getattr(message, "content", ""))


def _run_analysis_job(job_id: str, payload: dict, q: Queue):
    """Execute the MANTRA analysis pipeline, pushing updates to the SSE queue."""
    try:
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        from tradingagents.agents.utils.social_data_tools import set_finbert_status_callback
        from cli.stats_handler import StatsCallbackHandler

        # Build config from payload
        config = DEFAULT_CONFIG.copy()
        config["max_debate_rounds"] = payload.get("research_depth", 1)
        config["max_risk_discuss_rounds"] = payload.get("research_depth", 1)
        config["deep_think_provider"] = payload.get("manager_provider", "anthropic")
        config["quick_think_provider"] = payload.get("analyst_provider", "google")
        config["llm_provider"] = payload.get("analyst_provider", "google")
        config["deep_think_llm"] = payload.get("manager_model", "claude-opus-4-6")
        config["quick_think_llm"] = payload.get("analyst_model", "gemini-3-flash-preview")
        config["backend_url"] = payload.get("backend_url")
        config["google_thinking_level"] = payload.get("google_thinking_level")
        config["openai_reasoning_effort"] = payload.get("openai_reasoning_effort")
        config["anthropic_effort"] = payload.get("anthropic_effort")
        config["output_language"] = payload.get("output_language", "English")
        config["parallel_analysts"] = payload.get("parallel_analysts")
        # Fresh-run checkbox: when set, ignore any cached completed record and
        # force the pipeline to re-execute.
        if payload.get("fresh_run"):
            config["reuse_cached_run"] = False

        run_mode = payload.get("run_mode", "single")
        tickers = payload.get("tickers", ["SPY"])
        if isinstance(tickers, str):
            tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        ticker = tickers[0]
        start_date = payload.get("start_date", datetime.datetime.now().strftime("%Y-%m-%d"))
        end_date = payload.get("end_date", start_date)

        # Track run-record location on the job dict so /api/run_record can serve
        # the right ticker/date while the job progresses.
        job = _jobs.get(job_id)
        if job is not None:
            job["runs_dir"] = config.get("runs_dir", RUNS_DIR)
            job["ticker"] = ticker
            job["current_date"] = start_date

        selected_analysts = payload.get("analysts", ["market", "news", "fundamentals"])

        # Normalize analyst order
        selected_analyst_keys = [a for a in ANALYST_ORDER if a in selected_analysts]

        # Build initial agent status (shared with the CLI via tradingagents.streaming).
        agent_status = build_initial_agent_status(selected_analyst_keys, FIXED_AGENTS)
        debate_meta = {}  # tracks debate round numbers

        q.put(("status", {
            "agents": agent_status,
            "run_mode": run_mode,
            "parallel_analysts": None,  # determined after graph init
        }))

        if run_mode != "single":
            # Backtest mode
            config["run_mode"] = "backtest"
            _run_backtest_job(job_id, payload, config, tickers, selected_analyst_keys,
                             start_date, end_date, agent_status, q)
            return

        # Single-day analysis
        from tradingagents.observability import (
            RunRecorder, config_digest, find_replayable, set_recorder,
        )

        stats_handler = StatsCallbackHandler()

        graph = TradingAgentsGraph(
            selected_analyst_keys,
            config=config,
            debug=True,
            callbacks=[stats_handler],
        )

        # --- Replay / cache short-circuit ------------------------------------
        # When reuse is enabled and a completed record with a matching config
        # digest already exists, serve it instantly (mirrors propagate()'s
        # behaviour) and tell the frontend it was loaded from cache.
        runs_dir = config.get("runs_dir", RUNS_DIR)
        digest = config_digest(config)
        if config.get("reuse_cached_run", True):
            cached = find_replayable(runs_dir, ticker, start_date, digest)
            if cached:
                _emit_replayed_complete(q, agent_status, cached, ticker, start_date)
                return

        # Register FinBERT callback
        if "social" in selected_analyst_keys:
            set_finbert_status_callback(
                lambda status: q.put(("status", {"agent": "Media Labeling", "state": status}))
            )

        start_time = time.time()

        # Activate the run recorder so the per-agent monitor / metrics / final
        # report get written to runs/{ticker}/{date}.json by the pipeline hooks.
        recorder = RunRecorder(
            root=runs_dir, ticker=ticker, date=start_date,
            digest=digest, enabled=config.get("record_runs", True),
        )
        set_recorder(recorder)

        # Create result directory
        results_dir = Path(config["results_dir"]) / ticker / start_date
        results_dir.mkdir(parents=True, exist_ok=True)
        report_dir = results_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        # Mark initial analysts as in_progress
        is_parallel = graph._parallel_analysts
        if is_parallel:
            for key in selected_analyst_keys:
                agent_status[ANALYST_AGENT_NAMES[key]] = "in_progress"
        else:
            first_name = ANALYST_AGENT_NAMES[selected_analyst_keys[0]]
            agent_status[first_name] = "in_progress"
        q.put(("status", {
            "agents": agent_status,
            "run_mode": run_mode,
            "parallel_analysts": is_parallel,
        }))

        q.put(("message", {"type": "System", "content": f"Analyzing {ticker} on {start_date}..."}))

        # Initialize state
        init_state = graph.propagator.create_initial_state(ticker, start_date)
        args = graph.propagator.get_graph_args(callbacks=[stats_handler])
        graph.memory_store.set_analysis_date(start_date)

        report_sections: dict[str, str | None] = {}
        last_message_id = None

        # Stream analysis
        trace = []
        for chunk in graph.graph.stream(init_state, **args):
            # Process messages
            if len(chunk.get("messages", [])) > 0:
                last_message = chunk["messages"][-1]
                msg_id = getattr(last_message, "id", None)

                if msg_id != last_message_id:
                    last_message_id = msg_id
                    msg_type, content = _classify_message_type(last_message)
                    if content and content.strip():
                        q.put(("message", {"type": msg_type, "content": content[:500]}))

                    # Tool calls
                    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        for tc in last_message.tool_calls:
                            name = tc["name"] if isinstance(tc, dict) else tc.name
                            q.put(("tool", {"name": name}))

            # Update analyst statuses from report sections
            for section_key, (analyst_key, agent_name) in REPORT_SECTIONS.items():
                if analyst_key and analyst_key not in selected_analyst_keys:
                    continue
                if section_key in chunk and chunk[section_key]:
                    content = chunk[section_key]
                    if isinstance(content, list):
                        content = "\n".join(str(x) for x in content)
                    report_sections[section_key] = content
                    if analyst_key:
                        agent_status[agent_name] = "completed"
                        # In sequential mode, mark the next analyst as in_progress
                        if not is_parallel and analyst_key in selected_analyst_keys:
                            idx = selected_analyst_keys.index(analyst_key)
                            if idx + 1 < len(selected_analyst_keys):
                                next_key = selected_analyst_keys[idx + 1]
                                next_name = ANALYST_AGENT_NAMES[next_key]
                                agent_status[next_name] = "in_progress"
                            else:
                                # All analysts done, research team starts
                                agent_status["Bull Researcher"] = "in_progress"
                                agent_status["Bear Researcher"] = "in_progress"
                        q.put(("report", {"section": section_key, "content": content[:2000]}))

            # Research Team
            if chunk.get("investment_debate_state"):
                ds = chunk["investment_debate_state"]
                if ds.get("bull_history", "").strip() or ds.get("bear_history", "").strip():
                    agent_status["Bull Researcher"] = "in_progress"
                    agent_status["Bear Researcher"] = "in_progress"
                if ds.get("bull_history", "").strip():
                    agent_status["Bull Researcher"] = "completed"
                    q.put(("report", {"section": "bull_research", "content": ds["bull_history"][:2000]}))
                if ds.get("bear_history", "").strip():
                    agent_status["Bear Researcher"] = "completed"
                    q.put(("report", {"section": "bear_research", "content": ds["bear_history"][:2000]}))
                # Judge evaluates Bull/Bear arguments
                if ds.get("judge_history", "").strip() or ds.get("judge_critique_bull", "").strip() or ds.get("judge_critique_bear", "").strip():
                    agent_status["Judge"] = "in_progress"
                # Track debate round number
                debate_round = ds.get("judge_count", 0)
                if debate_round:
                    debate_meta["research_round"] = int(debate_round)
                if ds.get("judge_decision", "").strip():
                    agent_status["Judge"] = "completed"
                    agent_status["Research Manager"] = "completed"
                    agent_status["Trader"] = "in_progress"
                    report_sections["investment_plan"] = ds["judge_decision"]
                    q.put(("report", {"section": "investment_plan", "content": ds["judge_decision"][:2000]}))

            # Trader
            if chunk.get("trader_investment_plan"):
                agent_status["Trader"] = "completed"
                agent_status["Aggressive Analyst"] = "in_progress"
                report_sections["trader_investment_plan"] = chunk["trader_investment_plan"]
                q.put(("report", {"section": "trader_plan", "content": str(chunk["trader_investment_plan"])[:2000]}))

            # Risk Management
            if chunk.get("risk_debate_state"):
                rs = chunk["risk_debate_state"]
                if rs.get("aggressive_history", "").strip():
                    agent_status["Aggressive Analyst"] = "in_progress"
                if rs.get("conservative_history", "").strip():
                    agent_status["Conservative Analyst"] = "in_progress"
                if rs.get("neutral_history", "").strip():
                    agent_status["Neutral Analyst"] = "in_progress"
                if rs.get("judge_decision", "").strip():
                    agent_status["Portfolio Manager"] = "in_progress"
                    agent_status["Aggressive Analyst"] = "completed"
                    agent_status["Conservative Analyst"] = "completed"
                    agent_status["Neutral Analyst"] = "completed"

            # Final trade decision
            if chunk.get("final_trade_decision"):
                agent_status["Portfolio Manager"] = "completed"
                report_sections["final_trade_decision"] = chunk["final_trade_decision"]

            # Emit combined status update
            elapsed = round(time.time() - start_time, 1)
            q.put(("status", {
                "agents": agent_status.copy(),
                "debate_meta": debate_meta.copy(),
                "stats": {
                    "elapsed": elapsed,
                    "llm_calls": stats_handler.llm_calls,
                    "tool_calls": stats_handler.tool_calls,
                    "input_tokens": stats_handler.tokens_in,
                    "output_tokens": stats_handler.tokens_out,
                },
                "db_stats": {
                    "reads": graph.memory_store.db_reads,
                    "writes": graph.memory_store.db_writes,
                },
            }))

            trace.append(chunk)

        # Final state
        final_state = trace[-1] if trace else {}
        decision = graph.process_signal(final_state.get("final_trade_decision", ""))

        # Mark all as completed
        for agent in agent_status:
            agent_status[agent] = "completed"

        # Build final report
        final_report_parts = []
        for section_key in REPORT_SECTION_ORDER:
            content = final_state.get(section_key) or report_sections.get(section_key)
            if content:
                if isinstance(content, list):
                    content = "\n".join(str(x) for x in content)
                final_report_parts.append(content)

        # Finalize the run record so it shows status "completed" with the final
        # signal/reports for the detail + final-report views.
        try:
            recorder.finish(
                signal=decision,
                reports={
                    k: (final_state.get(k) or report_sections.get(k) or "")
                    for k in (
                        "market_report", "sentiment_report",
                        "news_report", "fundamentals_report",
                        "investment_plan", "trader_investment_plan",
                    )
                },
                decision_text=final_state.get("final_trade_decision", ""),
            )
        except Exception:
            pass
        finally:
            set_recorder(None)

        # Save report
        try:
            report_file = report_dir / "full_report.md"
            with open(report_file, "w") as f:
                f.write("\n\n---\n\n".join(final_report_parts))
        except Exception:
            pass

        # "Save full report" checkbox (default ON): write the structured report
        # to results/{TICKER}_{date}/report.md in the user-specified layout.
        # final_trade_decision is rendered as the decision_text header, so it is
        # excluded from the body sections to avoid duplication.
        if payload.get("save_report", True):
            reports_for_md = {
                k: (final_state.get(k) or report_sections.get(k) or "")
                for k in REPORT_SECTION_ORDER if k != "final_trade_decision"
            }
            out_dir = Path(RESULTS_DIR) / f"{ticker}_{start_date}"
            _save_report_md(
                out_dir / "report.md",
                signal=decision,
                decision_text=final_state.get("final_trade_decision", ""),
                reports=reports_for_md,
            )

        elapsed = round(time.time() - start_time, 1)
        q.put(("complete", {
            "agents": agent_status,
            "decision": decision,
            "report": "\n\n---\n\n".join(final_report_parts),
            "replay": False,
            "stats": {
                "elapsed": elapsed,
                "llm_calls": stats_handler.llm_calls,
                "tool_calls": stats_handler.tool_calls,
                "input_tokens": stats_handler.tokens_in,
                "output_tokens": stats_handler.tokens_out,
            },
            "db_stats": {
                "reads": graph.memory_store.db_reads,
                "writes": graph.memory_store.db_writes,
            },
        }))

    except Exception as e:
        import traceback
        # Mark the active run record (if any) as failed and detach the recorder.
        try:
            from tradingagents.observability import get_recorder, set_recorder
            get_recorder().fail(str(e))
            set_recorder(None)
        except Exception:
            pass
        q.put(("error", {"message": str(e), "traceback": traceback.format_exc()}))
    finally:
        job = _jobs.get(job_id)
        if job is not None:
            job["status"] = "done"


def _emit_replayed_complete(q, agent_status, cached, ticker, date):
    """Emit a completion event for a run served from the replay cache.

    The cached record already contains the final signal, reports, and decision
    text, so no pipeline runs — we just rebuild the report and flag replay=True
    so the frontend can show the "loaded from cache" banner.
    """
    final = cached.get("final") or {}
    reports = final.get("reports") or {}
    for agent in agent_status:
        agent_status[agent] = "completed"

    final_report_parts = []
    for section_key in REPORT_SECTION_ORDER:
        content = reports.get(section_key)
        if section_key == "final_trade_decision":
            content = content or final.get("decision_text")
        if content:
            if isinstance(content, list):
                content = "\n".join(str(x) for x in content)
            final_report_parts.append(content)
    if not final_report_parts and final.get("decision_text"):
        final_report_parts.append(final["decision_text"])

    q.put(("status", {"agents": agent_status.copy()}))
    q.put(("message", {
        "type": "System",
        "content": f"Loaded {ticker} {date} from cached run (no pipeline executed).",
    }))
    q.put(("complete", {
        "agents": agent_status,
        "decision": final.get("signal", {}),
        "report": "\n\n---\n\n".join(final_report_parts),
        "replay": True,
        "replay_date": date,
    }))


def _serialize_backtest_eval(ev: dict) -> dict | None:
    """Turn an evaluate_backtest() dict into a JSON-safe table for the GUI.

    Produces formatted strategy-vs-buy&hold rows (total return, annualized,
    max drawdown, Sharpe, Sortino, volatility, directional accuracy, turnover)
    matching the metrics.md layout, plus the day count.
    """
    if not ev or not ev.get("n_days"):
        return None
    m = ev.get("metrics") or {}
    strat = m.get("strategy") or {}
    bh = m.get("buy_hold") or {}

    def pct(v):
        return f"{v * 100:+.2f}%" if isinstance(v, (int, float)) and v == v else "n/a"

    def pct_plain(v):
        return f"{v * 100:.2f}%" if isinstance(v, (int, float)) and v == v else "n/a"

    def num(v):
        return f"{v:.2f}" if isinstance(v, (int, float)) and v == v else "n/a"

    rows = [
        {"metric": "Total return", "strategy": pct(strat.get("total_return")),
         "buy_hold": pct(bh.get("total_return"))},
        {"metric": "Annualized return", "strategy": pct(strat.get("annualized_return")),
         "buy_hold": pct(bh.get("annualized_return"))},
        {"metric": "Max drawdown", "strategy": pct_plain(strat.get("max_drawdown")),
         "buy_hold": pct_plain(bh.get("max_drawdown"))},
        {"metric": "Sharpe", "strategy": num(strat.get("sharpe")),
         "buy_hold": num(bh.get("sharpe"))},
        {"metric": "Sortino", "strategy": num(strat.get("sortino")),
         "buy_hold": num(bh.get("sortino"))},
        {"metric": "Volatility (ann.)", "strategy": pct_plain(strat.get("volatility_annualized")),
         "buy_hold": pct_plain(bh.get("volatility_annualized"))},
    ]
    acc = ev.get("accuracy")
    rows.append({
        "metric": "Directional accuracy",
        "strategy": f"{acc * 100:.1f}%" if acc is not None else "n/a",
        "buy_hold": "—",
    })
    rows.append({
        "metric": "Turnover (sum |Δpos|)",
        "strategy": num(ev.get("turnover")),
        "buy_hold": "—",
    })
    return {"rows": rows, "n_days": ev.get("n_days")}


def _eval_to_payload(ev: dict) -> dict:
    """Serialize a full evaluate_backtest() dict for the interactive GUI.

    Returns the formatted metrics ``table`` plus the raw equity series
    (``dates``/``strategy``/``buy_hold`` arrays) so the client can draw the
    comparison chart itself, along with the scalar accuracy/turnover/n_days.
    """
    table = _serialize_backtest_eval(ev)
    eq = ev.get("equity")
    dates: list = []
    strategy: list = []
    buy_hold: list = []
    try:
        if eq is not None and not eq.empty:
            dates = [str(d) for d in eq["date"].tolist()]
            strategy = [float(x) for x in eq["strategy"].tolist()]
            buy_hold = [float(x) for x in eq["buy_hold"].tolist()]
    except Exception:
        dates, strategy, buy_hold = [], [], []
    acc = ev.get("accuracy")
    return {
        "table": (table or {}).get("rows", []),
        "equity": {"dates": dates, "strategy": strategy, "buy_hold": buy_hold},
        "accuracy": acc,
        "turnover": ev.get("turnover"),
        "n_days": ev.get("n_days"),
    }


def _run_backtest_job(job_id, payload, config, tickers, selected_analyst_keys,
                      start_date, end_date, agent_status, q):
    """Run backtest mode in background thread with per-day streaming."""
    try:
        import pandas as pd
        import yfinance as yf
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        from tradingagents.dataflows.backtest_cache import get_backtest_cache
        from tradingagents.agents.utils.social_data_tools import set_finbert_status_callback
        from tradingagents.observability import (
            RunRecorder, config_digest, set_recorder,
        )
        from cli.stats_handler import StatsCallbackHandler

        start_time = time.time()
        stats_handler = StatsCallbackHandler()
        cache = get_backtest_cache()
        runs_dir = config.get("runs_dir", RUNS_DIR)
        digest = config_digest(config)

        all_results = []
        # Per-ticker signal rows are retained on the job so the interactive
        # completion view can recompute analytics with tunable parameters and
        # save them on demand (nothing is auto-saved at completion).
        rows_by_ticker: dict[str, list] = {}

        for ticker in tickers:
            rows_by_ticker.setdefault(ticker, [])
            q.put(("message", {"type": "System", "content": f"Pre-fetching data cache for {ticker}..."}))
            cache.initialize(ticker, start_date, end_date)

            # Determine trading days
            try:
                hist = yf.download(
                    ticker, start=start_date, end=end_date,
                    auto_adjust=True, progress=False,
                )
                trading_days = [d.strftime("%Y-%m-%d") for d in hist.index]
            except Exception:
                trading_days = pd.bdate_range(start_date, end_date).strftime("%Y-%m-%d").tolist()

            if not trading_days:
                q.put(("message", {"type": "System", "content": f"No trading days found for {ticker}"}))
                continue

            q.put(("message", {"type": "System", "content": f"Starting backtest for {ticker}: {len(trading_days)} trading days"}))

            graph = TradingAgentsGraph(
                selected_analyst_keys,
                config=config,
                debug=True,
                callbacks=[stats_handler],
            )

            is_parallel = graph._parallel_analysts

            # Register FinBERT callback
            if "social" in selected_analyst_keys:
                set_finbert_status_callback(
                    lambda status: q.put(("status", {"agent": "Media Labeling", "state": status}))
                )

            # Surface the current ticker for /api/run_record while this ticker runs.
            _job = _jobs.get(job_id)
            if _job is not None:
                _job["ticker"] = ticker

            for day_idx, trade_date in enumerate(trading_days):
                # --- Reset agent statuses for this day ---
                for a in agent_status:
                    agent_status[a] = "pending"

                # Advance the served run-record date as the backtest walks days.
                if _job is not None:
                    _job["current_date"] = trade_date

                # Activate a fresh run recorder for this day so the per-agent
                # monitor / metrics / report views reflect the current day.
                recorder = RunRecorder(
                    root=runs_dir, ticker=ticker, date=trade_date,
                    digest=digest, enabled=config.get("record_runs", True),
                )
                set_recorder(recorder)

                # Send backtest progress
                q.put(("backtest_progress", {
                    "ticker": ticker,
                    "current_date": trade_date,
                    "day_index": day_idx + 1,
                    "total_days": len(trading_days),
                    "start_date": trading_days[0],
                    "end_date": trading_days[-1],
                }))

                # Mark initial analysts
                if is_parallel:
                    for key in selected_analyst_keys:
                        agent_status[ANALYST_AGENT_NAMES[key]] = "in_progress"
                else:
                    first_name = ANALYST_AGENT_NAMES[selected_analyst_keys[0]]
                    agent_status[first_name] = "in_progress"

                q.put(("status", {
                    "agents": agent_status.copy(),
                    "parallel_analysts": is_parallel,
                }))
                q.put(("message", {"type": "System", "content": f"[Day {day_idx+1}/{len(trading_days)}] Analyzing {ticker} on {trade_date}..."}))

                # Initialize state for this day
                graph.memory_store.set_analysis_date(trade_date)
                init_state = graph.propagator.create_initial_state(ticker, trade_date)
                args = graph.propagator.get_graph_args(callbacks=[stats_handler])

                debate_meta = {}
                last_message_id = None
                trace = []

                # Stream the graph for this day
                for chunk in graph.graph.stream(init_state, **args):
                    # Process messages
                    if len(chunk.get("messages", [])) > 0:
                        last_message = chunk["messages"][-1]
                        msg_id = getattr(last_message, "id", None)
                        if msg_id != last_message_id:
                            last_message_id = msg_id
                            msg_type, content = _classify_message_type(last_message)
                            if content and content.strip():
                                q.put(("message", {"type": msg_type, "content": content[:500]}))
                            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                                for tc in last_message.tool_calls:
                                    name = tc["name"] if isinstance(tc, dict) else tc.name
                                    q.put(("tool", {"name": name}))

                    # Update analyst statuses from report sections
                    for section_key, (analyst_key, agent_name) in REPORT_SECTIONS.items():
                        if analyst_key and analyst_key not in selected_analyst_keys:
                            continue
                        if section_key in chunk and chunk[section_key]:
                            content = chunk[section_key]
                            if isinstance(content, list):
                                content = "\n".join(str(x) for x in content)
                            if analyst_key:
                                agent_status[agent_name] = "completed"
                                if not is_parallel and analyst_key in selected_analyst_keys:
                                    idx = selected_analyst_keys.index(analyst_key)
                                    if idx + 1 < len(selected_analyst_keys):
                                        next_key = selected_analyst_keys[idx + 1]
                                        agent_status[ANALYST_AGENT_NAMES[next_key]] = "in_progress"
                                    else:
                                        agent_status["Bull Researcher"] = "in_progress"
                                        agent_status["Bear Researcher"] = "in_progress"
                                # Don't accumulate reports across days in backtest
                                q.put(("report", {"section": section_key, "content": content[:2000]}))

                    # Research Team
                    if chunk.get("investment_debate_state"):
                        ds = chunk["investment_debate_state"]
                        if ds.get("bull_history", "").strip() or ds.get("bear_history", "").strip():
                            agent_status["Bull Researcher"] = "in_progress"
                            agent_status["Bear Researcher"] = "in_progress"
                        if ds.get("bull_history", "").strip():
                            agent_status["Bull Researcher"] = "completed"
                        if ds.get("bear_history", "").strip():
                            agent_status["Bear Researcher"] = "completed"
                        if ds.get("judge_history", "").strip() or ds.get("judge_critique_bull", "").strip() or ds.get("judge_critique_bear", "").strip():
                            agent_status["Judge"] = "in_progress"
                        debate_round = ds.get("judge_count", 0)
                        if debate_round:
                            debate_meta["research_round"] = int(debate_round)
                        if ds.get("judge_decision", "").strip():
                            agent_status["Judge"] = "completed"
                            agent_status["Research Manager"] = "completed"
                            agent_status["Trader"] = "in_progress"

                    # Trader
                    if chunk.get("trader_investment_plan"):
                        agent_status["Trader"] = "completed"
                        agent_status["Aggressive Analyst"] = "in_progress"

                    # Risk Management
                    if chunk.get("risk_debate_state"):
                        rs = chunk["risk_debate_state"]
                        if rs.get("aggressive_history", "").strip():
                            agent_status["Aggressive Analyst"] = "in_progress"
                        if rs.get("conservative_history", "").strip():
                            agent_status["Conservative Analyst"] = "in_progress"
                        if rs.get("neutral_history", "").strip():
                            agent_status["Neutral Analyst"] = "in_progress"
                        if rs.get("judge_decision", "").strip():
                            agent_status["Portfolio Manager"] = "in_progress"
                            agent_status["Aggressive Analyst"] = "completed"
                            agent_status["Conservative Analyst"] = "completed"
                            agent_status["Neutral Analyst"] = "completed"

                    # Final trade decision
                    if chunk.get("final_trade_decision"):
                        agent_status["Portfolio Manager"] = "completed"

                    # Emit combined status update
                    elapsed = round(time.time() - start_time, 1)
                    q.put(("status", {
                        "agents": agent_status.copy(),
                        "debate_meta": debate_meta.copy(),
                        "stats": {
                            "elapsed": elapsed,
                            "llm_calls": stats_handler.llm_calls,
                            "tool_calls": stats_handler.tool_calls,
                            "input_tokens": stats_handler.tokens_in,
                            "output_tokens": stats_handler.tokens_out,
                        },
                        "db_stats": {
                            "reads": graph.memory_store.db_reads,
                            "writes": graph.memory_store.db_writes,
                        },
                    }))

                    trace.append(chunk)

                # Day complete — process signal and reflect
                final_state = trace[-1] if trace else {}
                graph.curr_state = final_state
                graph.ticker = ticker
                signal_dict = graph.process_signal(final_state.get("final_trade_decision", ""))
                graph._last_signal_dict = signal_dict

                # Finalize this day's run record before moving on.
                try:
                    recorder.finish(
                        signal=signal_dict,
                        reports={
                            k: (final_state.get(k) or "")
                            for k in (
                                "market_report", "sentiment_report",
                                "news_report", "fundamentals_report",
                                "investment_plan", "trader_investment_plan",
                            )
                        },
                        decision_text=final_state.get("final_trade_decision", ""),
                    )
                except Exception:
                    pass
                finally:
                    set_recorder(None)

                # Save this day's final report to the backtest results folder.
                # (Intermediate steps live in runs/ — only the day's report here.)
                bt_out_dir = Path(RESULTS_DIR) / f"{ticker}_{start_date}_{end_date}"
                reports_for_md = {
                    k: (final_state.get(k) or "")
                    for k in REPORT_SECTION_ORDER if k != "final_trade_decision"
                }
                _save_report_md(
                    bt_out_dir / "daily_reports" / f"{trade_date}.md",
                    signal=signal_dict,
                    decision_text=final_state.get("final_trade_decision", ""),
                    reports=reports_for_md,
                )

                actual_return = cache.get_next_day_return(trade_date)
                graph.reflect_and_remember(actual_return)

                row = {
                    "ticker": ticker,
                    "date": trade_date,
                    "signal": signal_dict.get("signal", "HOLD"),
                    "confidence": signal_dict.get("confidence", 0.70),
                    "horizon": signal_dict.get("horizon", "1-5d"),
                    "actual_return": actual_return,
                }
                all_results.append(row)
                rows_by_ticker[ticker].append(row)

                q.put(("message", {
                    "type": "System",
                    "content": f"[Day {day_idx+1}/{len(trading_days)}] {ticker} {trade_date}: {signal_dict.get('signal', 'HOLD')} (return: {actual_return:+.4f})",
                }))

            cache.clear()
            q.put(("message", {
                "type": "System",
                "content": f"Completed {len(trading_days)} trading days for {ticker}",
            }))

        elapsed = round(time.time() - start_time, 1)

        # Summarize results
        summary_lines = []
        for r in all_results:
            summary_lines.append(
                f"{r.get('ticker', '?')} {r.get('date', '?')}: "
                f"{r.get('signal', '?')} (confidence: {r.get('confidence', '?')})"
            )

        # --- Backtest analytics: DEFERRED save ---------------------------------
        # Analytics are NOT written to disk at completion. The per-ticker signal
        # rows are retained on the job so the interactive completion view can
        # recompute the equity curve + metrics with user-tunable parameters
        # (/api/backtest_eval) and persist them on demand (/api/backtest_save).
        # The per-day daily_reports/{date}.md files were already saved above.
        _job = _jobs.get(job_id)
        if _job is not None:
            _job["backtest_rows"] = rows_by_ticker
            _job["backtest_start"] = start_date
            _job["backtest_end"] = end_date

        q.put(("complete", {
            "agents": {a: "completed" for a in agent_status},
            "decision": {"summary": "Backtest complete"},
            "report": "\n".join(summary_lines),
            "replay": False,
            "backtest": True,
            # Tickers available for the interactive analytics view; the client
            # fetches each one's chart + table from /api/backtest_eval.
            "backtest_tickers": list(rows_by_ticker.keys()),
            "backtest_start": start_date,
            "backtest_end": end_date,
            "stats": {
                "elapsed": elapsed,
                "llm_calls": stats_handler.llm_calls,
                "tool_calls": stats_handler.tool_calls,
                "input_tokens": stats_handler.tokens_in,
                "output_tokens": stats_handler.tokens_out,
            },
            "db_stats": {
                "reads": graph.memory_store.db_reads,
                "writes": graph.memory_store.db_writes,
            },
        }))

    except Exception as e:
        import traceback
        try:
            from tradingagents.observability import get_recorder, set_recorder
            get_recorder().fail(str(e))
            set_recorder(None)
        except Exception:
            pass
        q.put(("error", {"message": str(e), "traceback": traceback.format_exc()}))
    finally:
        job = _jobs.get(job_id)
        if job is not None:
            job["status"] = "done"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Entry point for the `mantragui` console script."""
    host = "0.0.0.0"
    port = 5720
    url = f"http://127.0.0.1:{port}"

    print(f"\n  MANTRA GUI starting at {url}\n")

    # Open browser after a short delay
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
