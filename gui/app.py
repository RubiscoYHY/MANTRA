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


# ---------------------------------------------------------------------------
# Analysis runner (runs in background thread)
# ---------------------------------------------------------------------------

# Agent display name mapping
ANALYST_AGENT_NAMES = {
    "market": "Market Analyst",
    "social": "Social Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
}

ANALYST_ORDER = ["market", "social", "news", "fundamentals"]

FIXED_AGENTS = {
    "Research Team": ["Bull Researcher", "Bear Researcher", "Judge", "Research Manager"],
    "Trading Team": ["Trader"],
    "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
    "Portfolio Management": ["Portfolio Manager"],
}

REPORT_SECTIONS = {
    "market_report": ("market", "Market Analyst"),
    "sentiment_report": ("social", "Social Analyst"),
    "news_report": ("news", "News Analyst"),
    "fundamentals_report": ("fundamentals", "Fundamentals Analyst"),
    "investment_plan": (None, "Research Manager"),
    "trader_investment_plan": (None, "Trader"),
    "final_trade_decision": (None, "Portfolio Manager"),
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

        run_mode = payload.get("run_mode", "single")
        tickers = payload.get("tickers", ["SPY"])
        if isinstance(tickers, str):
            tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        ticker = tickers[0]
        start_date = payload.get("start_date", datetime.datetime.now().strftime("%Y-%m-%d"))
        end_date = payload.get("end_date", start_date)

        selected_analysts = payload.get("analysts", ["market", "news", "fundamentals"])

        # Normalize analyst order
        selected_analyst_keys = [a for a in ANALYST_ORDER if a in selected_analysts]

        # Build initial agent status
        agent_status = {}
        debate_meta = {}  # tracks debate round numbers
        for key in selected_analyst_keys:
            agent_status[ANALYST_AGENT_NAMES[key]] = "pending"
        if "social" in selected_analyst_keys:
            agent_status["Media Labeling"] = "pending"
        for group_agents in FIXED_AGENTS.values():
            for agent_name in group_agents:
                agent_status[agent_name] = "pending"

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
        stats_handler = StatsCallbackHandler()

        graph = TradingAgentsGraph(
            selected_analyst_keys,
            config=config,
            debug=True,
            callbacks=[stats_handler],
        )

        # Register FinBERT callback
        if "social" in selected_analyst_keys:
            set_finbert_status_callback(
                lambda status: q.put(("status", {"agent": "Media Labeling", "state": status}))
            )

        start_time = time.time()

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
        for section_key in ["market_report", "sentiment_report", "news_report",
                            "fundamentals_report", "investment_plan",
                            "trader_investment_plan", "final_trade_decision"]:
            content = final_state.get(section_key) or report_sections.get(section_key)
            if content:
                if isinstance(content, list):
                    content = "\n".join(str(x) for x in content)
                final_report_parts.append(content)

        # Save report
        try:
            report_file = report_dir / "full_report.md"
            with open(report_file, "w") as f:
                f.write("\n\n---\n\n".join(final_report_parts))
        except Exception:
            pass

        elapsed = round(time.time() - start_time, 1)
        q.put(("complete", {
            "agents": agent_status,
            "decision": decision,
            "report": "\n\n---\n\n".join(final_report_parts),
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
        q.put(("error", {"message": str(e), "traceback": traceback.format_exc()}))


def _run_backtest_job(job_id, payload, config, tickers, selected_analyst_keys,
                      start_date, end_date, agent_status, q):
    """Run backtest mode in background thread with per-day streaming."""
    try:
        import pandas as pd
        import yfinance as yf
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        from tradingagents.dataflows.backtest_cache import get_backtest_cache
        from tradingagents.agents.utils.social_data_tools import set_finbert_status_callback
        from cli.stats_handler import StatsCallbackHandler

        start_time = time.time()
        stats_handler = StatsCallbackHandler()
        cache = get_backtest_cache()

        all_results = []

        for ticker in tickers:
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

            for day_idx, trade_date in enumerate(trading_days):
                # --- Reset agent statuses for this day ---
                for a in agent_status:
                    agent_status[a] = "pending"

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

                actual_return = cache.get_next_day_return(trade_date)
                graph.reflect_and_remember(actual_return)

                all_results.append({
                    "ticker": ticker,
                    "date": trade_date,
                    "signal": signal_dict.get("signal", "HOLD"),
                    "confidence": signal_dict.get("confidence", 0.70),
                    "horizon": signal_dict.get("horizon", "1-5d"),
                    "actual_return": actual_return,
                })

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

        q.put(("complete", {
            "agents": {a: "completed" for a in agent_status},
            "decision": {"summary": "Backtest complete"},
            "report": "\n".join(summary_lines),
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
        q.put(("error", {"message": str(e), "traceback": traceback.format_exc()}))


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
