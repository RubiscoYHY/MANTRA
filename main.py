"""
MANTRA — Python API entry point
================================
Use this file to run single-day analysis or a multi-day backtest loop directly
from Python (e.g., in a Jupyter notebook or Google Colab).

Quick start
-----------
1. Install dependencies:
       pip install -e .

2. Set API keys (choose one method):

   Method A — .env file (local development):
       Create a .env file in the project root:
           ANTHROPIC_API_KEY=sk-ant-...
           GOOGLE_API_KEY=AIza...

   Method B — environment variables (Google Colab / CI):
       import os
       os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
       os.environ["GOOGLE_API_KEY"]    = "AIza..."

       # Or use Colab secrets:
       # from google.colab import userdata
       # os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
       # os.environ["GOOGLE_API_KEY"]    = userdata.get("GOOGLE_API_KEY")

Return value of propagate()
----------------------------
ta.propagate(ticker, date) returns (final_state, signal_dict) where:
    signal_dict = {
        "signal":     "BUY" | "OVERWEIGHT" | "HOLD" | "UNDERWEIGHT" | "SELL",
        "confidence": float,   # [0.50, 1.00]
        "horizon":    str,     # "1-5d" | "5-20d" | "20d+"
    }
"""

import os
from dotenv import load_dotenv

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# Load API keys from .env (no-op if the file does not exist, e.g. on Colab)
load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
config = DEFAULT_CONFIG.copy()

# Analyst layer (Market / Social / News / Fundamentals analysts, Researchers, Trader)
# Option A — Google Gemini Flash (recommended; fast & cost-effective)
config["quick_think_provider"] = "google"            # Requires: GOOGLE_API_KEY
config["quick_think_llm"]      = "gemini-2.5-flash"

# Option B — Ollama local model (no API key required)
# config["quick_think_provider"] = "ollama"
# config["quick_think_llm"]      = "gemma4:latest"
# config["backend_url"]          = "http://localhost:11434/v1"

# Manager layer (Research Manager / Portfolio Manager — final decision nodes)
# Option A — Anthropic Claude Opus (recommended for best reasoning quality)
config["deep_think_provider"]  = "anthropic"         # Requires: ANTHROPIC_API_KEY
config["deep_think_llm"]       = "claude-opus-4-6"

# Option B — same model as analysts
# config["deep_think_provider"] = "google"
# config["deep_think_llm"]      = "gemini-2.5-flash"

# Optional settings
config["max_debate_rounds"]    = 1      # increase for deeper bull/bear debate
config["output_language"]      = "English"  # language for analyst reports
config["results_dir"]          = "./results"  # output directory (auto-created)

# ── Single-day analysis ────────────────────────────────────────────────────────
ta = TradingAgentsGraph(debug=True, config=config)

_, signal_dict = ta.propagate("NVDA", "2024-05-10")

print("Signal:    ", signal_dict["signal"])
print("Confidence:", signal_dict["confidence"])
print("Horizon:   ", signal_dict["horizon"])

# Reflect on the decision after observing the actual return
# (pass the realised P&L; positive = profit, negative = loss)
# ta.reflect_and_remember(1000)

# ── Multi-day backtest loop (Python API) ───────────────────────────────────────
# There is no built-in run_backtest() method; write a loop like this:
#
# import pandas as pd
#
# config["run_mode"] = "backtest"   # enables persistent memory across days
# ta = TradingAgentsGraph(debug=False, config=config)
#
# trade_dates = pd.bdate_range("2024-01-02", "2024-03-29")
# results = []
#
# for i, date in enumerate(trade_dates):
#     date_str = date.strftime("%Y-%m-%d")
#     _, signal_dict = ta.propagate("AAPL", date_str)
#
#     # Compute next-day return (requires price data)
#     # actual_return = (price[i+1] - price[i]) / price[i]
#     actual_return = 0.0  # replace with real value
#
#     ta.reflect_and_remember(actual_return)
#     results.append({"date": date_str, **signal_dict, "actual_return": actual_return})
#
# import pandas as pd
# df = pd.DataFrame(results)
# df.to_csv("results/backtest_AAPL.csv", index=False)
#
# # Visualise with backtest_analyze
# from tradingagents.graph.backtest_analyze import _build_figure, _download_ohlcv
# ohlcv = _download_ohlcv("AAPL", "2024-01-02", "2024-04-01")
# fig, values = _build_figure(df=df, ticker="AAPL", ohlcv=ohlcv,
#                             initial_capital=10000.0, threshold=0.65, rf_annual=0.05)
# fig.savefig("results/AAPL_analysis.png", dpi=150, bbox_inches="tight")
