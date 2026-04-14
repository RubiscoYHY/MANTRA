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

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Tuple

from langchain_core.messages import ToolMessage


# ---------------------------------------------------------------------------
# Analyst layer
# ---------------------------------------------------------------------------

def _run_single_analyst_react(
    analyst_node_fn: Callable,
    tool_node: Any,
    initial_messages: List,
    state_context: Dict,
) -> Dict:
    """Run one analyst's full ReAct loop with an isolated message history.

    Replicates the LangGraph conditional-edge loop (analyst ↔ tool_node)
    without touching shared state.  Uses the existing ToolNode so all
    vendor-routing and caching logic is preserved.

    Returns only the report fields (e.g. ``{"market_report": "..."}``);
    the ``messages`` key is stripped so the caller's AgentState.messages
    remains unchanged.
    """
    messages = list(initial_messages)

    while True:
        local_state = {**state_context, "messages": messages}
        result = analyst_node_fn(local_state)

        new_messages = result.get("messages", [])
        if new_messages:
            messages = messages + new_messages

        last_msg = messages[-1]
        if not (hasattr(last_msg, "tool_calls") and last_msg.tool_calls):
            # No pending tool calls — LLM has produced its final report.
            return {k: v for k, v in result.items() if k != "messages"}

        # Execute tool calls directly — ToolNode.invoke() requires the
        # LangGraph runtime config context which is not propagated to worker
        # threads.  Calling the underlying tools directly avoids this.
        tool_messages = []
        for tc in last_msg.tool_calls:
            tool = tool_node.tools_by_name.get(tc["name"])
            if tool is None:
                continue
            try:
                result = tool.invoke(tc["args"])
            except Exception as exc:
                result = f"Error: {exc}"
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
) -> Dict:
    """Run all selected analysts concurrently and merge their report fields.

    Args:
        analyst_configs:  [(analyst_type, node_fn, tool_node), ...] for every
                          analyst that should run this cycle.
        initial_messages: Snapshot of AgentState.messages at entry (used as
                          each analyst's starting context; each gets its own
                          copy so their histories never interfere).
        state_context:    Shared state fields required by every analyst node
                          (``company_of_interest``, ``trade_date``, …).

    Returns:
        Merged dict of all report fields
        (``market_report``, ``sentiment_report``, ``news_report``,
        ``fundamentals_report`` — whichever analysts were selected).
    """
    merged: Dict = {}
    with ThreadPoolExecutor(max_workers=len(analyst_configs)) as executor:
        future_to_type = {
            executor.submit(
                _run_single_analyst_react,
                node_fn,
                tool_node,
                list(initial_messages),   # isolated copy per analyst
                dict(state_context),
            ): analyst_type
            for analyst_type, node_fn, tool_node in analyst_configs
        }
        for future in as_completed(future_to_type):
            # Re-raises any exception that occurred in the worker thread.
            merged.update(future.result())
    return merged


