"""
tradingagents/agents/researchers/researcher_round.py

Parallel Researcher Round node.

Runs Bull and Bear analysts concurrently via ThreadPoolExecutor. Neither
analyst sees the other's output — each only sees its own history plus the
Judge's latest directive (or nothing, on the first round).

Graph flow:
    [Analyst layer] → Researcher Round ──→ Judge Researcher
                              ↑                    │ (static edge)
                              └────────────────────┘
                      (loops until judge_count >= judge_iterations)
                      Then: Researcher Round → Research Manager
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from tradingagents.agents.researchers.bull_researcher import _build_bull_argument
from tradingagents.agents.researchers.bear_researcher import _build_bear_argument

logger = logging.getLogger(__name__)


def create_researcher_round(llm, memory_store):
    """
    Create the parallel Researcher Round LangGraph node.

    Args:
        llm:          LLM instance for both researchers (quick-thinking tier).
        memory_store: TradingMemoryStore for memory retrieval.

    Returns:
        A LangGraph-compatible node function that runs Bull and Bear in
        parallel and merges their outputs into a single state update.
    """

    def researcher_round_node(state: dict) -> dict:
        debate_state = state["investment_debate_state"]
        judge_count  = debate_state.get("judge_count", 0)

        logger.debug(
            "Researcher Round starting (judge_count=%d, ticker=%s)",
            judge_count,
            state.get("company_of_interest", "?"),
        )

        # Run Bull and Bear in parallel; each only reads its own history
        # and the Judge's directive addressed specifically to it.
        with ThreadPoolExecutor(max_workers=2) as executor:
            bull_future = executor.submit(
                _build_bull_argument, state, llm, memory_store
            )
            bear_future = executor.submit(
                _build_bear_argument, state, llm, memory_store
            )
            # Collect results; propagate exceptions from threads
            bull_content = bull_future.result()
            bear_content = bear_future.result()

        bull_turn = f"Bull Analyst: {bull_content}"
        bear_turn = f"Bear Analyst: {bear_content}"

        round_block = f"\n--- Researcher Round (after Judge iteration {judge_count}) ---\n{bull_turn}\n{bear_turn}\n"

        new_debate_state = {
            "bull_history":      debate_state.get("bull_history",      "") + "\n" + bull_turn,
            "bear_history":      debate_state.get("bear_history",      "") + "\n" + bear_turn,
            "history":           debate_state.get("history",           "") + round_block,
            # Judge fields carried forward unchanged
            "judge_history":     debate_state.get("judge_history",     ""),
            "judge_critique_bull": debate_state.get("judge_critique_bull", ""),
            "judge_critique_bear": debate_state.get("judge_critique_bear", ""),
            "judge_count":       judge_count,
            # Legacy fields
            "current_response":  bear_turn,
            "judge_decision":    debate_state.get("judge_decision",    ""),
            "count":             debate_state.get("count",              0) + 2,
        }

        return {"investment_debate_state": new_debate_state}

    return researcher_round_node
