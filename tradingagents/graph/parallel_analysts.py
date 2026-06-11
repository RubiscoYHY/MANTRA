# tradingagents/graph/parallel_analysts.py
"""Concurrent execution helpers for the analyst layer.

Each of the four analysts (market, social, news, fundamentals) maintains its
own isolated message history and runs its full ReAct loop (LLM ↔ ToolNode)
inside a dedicated thread.  Only the final report fields are written back to
the shared AgentState; the shared ``messages`` list is left untouched.

The risk-analyst layer (aggressive / conservative / neutral) retains the
original sequential rotation so that each analyst can immediately rebut the
previous speaker's argument within the same debate round.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Tuple

from langchain_core.messages import ToolMessage

from tradingagents.observability import count_news_items, get_recorder
from tradingagents.streaming import ANALYST_AGENT_NAMES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Analyst layer
# ---------------------------------------------------------------------------

def _run_single_analyst_react(
    analyst_node_fn: Callable,
    tool_node: Any,
    initial_messages: List,
    state_context: Dict,
    agent_name: str = "",
) -> Dict:
    """Run one analyst's full ReAct loop with an isolated message history.

    Replicates the LangGraph conditional-edge loop (analyst ↔ tool_node)
    without touching shared state.  Uses the existing ToolNode so all
    vendor-routing and caching logic is preserved.

    Every LLM turn (with the tool names it requested) and every tool result
    is recorded to the active RunRecorder for the agent monitor views.

    Returns only the report fields (e.g. ``{"market_report": "..."}``);
    the ``messages`` key is stripped so the caller's AgentState.messages
    remains unchanged.
    """
    rec = get_recorder()
    messages = list(initial_messages)

    while True:
        local_state = {**state_context, "messages": messages}
        result = analyst_node_fn(local_state)

        new_messages = result.get("messages", [])
        if new_messages:
            messages = messages + new_messages

        last_msg = messages[-1]
        pending_calls = getattr(last_msg, "tool_calls", None) or []
        rec.llm_turn(
            agent_name,
            getattr(last_msg, "content", ""),
            [tc["name"] for tc in pending_calls],
        )

        if not pending_calls:
            # No pending tool calls — LLM has produced its final report.
            return {k: v for k, v in result.items() if k != "messages"}

        # Execute tool calls directly — ToolNode.invoke() requires the
        # LangGraph runtime config context which is not propagated to worker
        # threads.  Calling the underlying tools directly avoids this.
        tool_messages = []
        for tc in pending_calls:
            tool = tool_node.tools_by_name.get(tc["name"])
            if tool is None:
                continue
            try:
                result = tool.invoke(tc["args"])
            except Exception as exc:
                # The error string goes back into the ReAct loop so the LLM
                # can adapt, but it must also be visible to the operator.
                logger.error("Tool %s failed: %s", tc["name"], exc, exc_info=True)
                result = f"Error: {exc}"
            rec.tool_call(agent_name, tc["name"], tc.get("args") or {}, result)
            if "news" in tc["name"]:
                rec.bump_metric("news", tc["name"], count_news_items(result))
            tool_messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tc["id"],
                    name=tc["name"],
                )
            )
        messages = messages + tool_messages


def run_analysts_parallel(
    analyst_configs: List[Tuple[str, Callable, Any]],
    initial_messages: List,
    state_context: Dict,
    max_workers: int | None = None,
) -> Dict:
    """Run all selected analysts and merge their report fields.

    This is the single analyst-execution path for both modes: cloud providers
    run with one thread per analyst, local providers (hardware-limited) pass
    max_workers=1 and get strictly sequential execution through the exact
    same code.

    Args:
        analyst_configs:  [(analyst_type, node_fn, tool_node), ...] for every
                          analyst that should run this cycle.
        initial_messages: Snapshot of AgentState.messages at entry (used as
                          each analyst's starting context; each gets its own
                          copy so their histories never interfere).
        state_context:    Shared state fields required by every analyst node
                          (``company_of_interest``, ``trade_date``, …).
        max_workers:      Thread pool size; defaults to one per analyst.
                          Use 1 for sequential execution (local models).

    Returns:
        Merged dict of all report fields
        (``market_report``, ``sentiment_report``, ``news_report``,
        ``fundamentals_report`` — whichever analysts were selected).
    """
    rec = get_recorder()
    merged: Dict = {}
    workers = max_workers or len(analyst_configs)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_type = {}
        for analyst_type, node_fn, tool_node in analyst_configs:
            agent_name = ANALYST_AGENT_NAMES.get(analyst_type, analyst_type)
            rec.agent_start(agent_name, inputs=dict(state_context))
            future = executor.submit(
                _run_single_analyst_react,
                node_fn,
                tool_node,
                list(initial_messages),   # isolated copy per analyst
                dict(state_context),
                agent_name,
            )
            future_to_type[future] = (analyst_type, agent_name)
        for future in as_completed(future_to_type):
            # Re-raises any exception that occurred in the worker thread.
            result = future.result()
            _, agent_name = future_to_type[future]
            rec.agent_finish(agent_name, output=result)
            merged.update(result)
    return merged


