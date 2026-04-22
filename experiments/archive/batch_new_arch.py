"""
experiments/batch_new_arch.py

Run 10 tickers on 2026-04-18 using the NEW Judge-mediated debate architecture.

Config: judge_iterations=1  →  Bull+Bear parallel (round 0)
                              → Judge critique
                              → Bull+Bear parallel (round 1)
                              → Research Manager decides.
"""

import json
import traceback
from datetime import date

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

import sys as _sys
TICKER_GROUPS = {
    "original": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "JPM", "XOM", "JNJ", "META", "TSLA"],
    "new":      ["AVGO", "TSM", "BRK.B", "WMT", "LLY", "V", "MU", "ORCL", "MA", "AMD"],
}
_date_str  = _sys.argv[1] if len(_sys.argv) > 1 else "2026-04-18"
_group     = _sys.argv[2] if len(_sys.argv) > 2 else "original"
TRADE_DATE = date.fromisoformat(_date_str)
TICKERS    = TICKER_GROUPS[_group]
OUTPUT_PATH = f"experiments/results_new_arch_{_group}_{_date_str}.json"

CONFIG = {
    **DEFAULT_CONFIG,
    # All reasoning via Google Gemini
    "llm_provider":         "google",
    "deep_think_provider":  "google",
    "deep_think_llm":       "gemini-3.1-pro-preview",
    "quick_think_provider": "google",
    "quick_think_llm":      "gemini-3.1-flash-lite-preview",
    # Shallowest judge loop: 1 judge iteration
    "judge_iterations":     1,
    "max_risk_discuss_rounds": 1,
    "parallel_analysts":    True,
}

results = []

for ticker in TICKERS:
    print(f"\n{'='*50}")
    print(f"NEW ARCH  |  {ticker}  |  {TRADE_DATE}")
    print(f"{'='*50}")
    try:
        ta = TradingAgentsGraph(config=CONFIG)
        final_state, signal_dict = ta.propagate(ticker, TRADE_DATE)

        debate = final_state["investment_debate_state"]

        row = {
            "arch":           "new",
            "ticker":         ticker,
            "signal":         signal_dict.get("signal", "UNKNOWN"),
            "confidence":     signal_dict.get("confidence", None),
            "judge_count":    debate.get("judge_count", 0),
            "bull_len":       len(debate.get("bull_history", "")),
            "bear_len":       len(debate.get("bear_history", "")),
            "judge_decision": debate.get("judge_decision", ""),
        }
        results.append(row)
        print(f"  RESULT: signal={row['signal']}  confidence={row['confidence']}")

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results.append({"arch": "new", "ticker": ticker, "error": str(e)})

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n\nResults saved to {OUTPUT_PATH}")

signals = [r.get("signal", "ERROR") for r in results if "error" not in r]
print("\n=== NEW ARCH SIGNAL SUMMARY ===")
for sig in ["BUY", "SELL", "HOLD"]:
    print(f"  {sig}: {signals.count(sig)}/{len(signals)}")
