# TradingAgents/graph/setup.py

from typing import Any, Dict
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.agents.researchers.researcher_round import create_researcher_round
from tradingagents.agents.researchers.judge_researcher import create_judge_researcher
from tradingagents.observability import get_recorder

from .conditional_logic import ConditionalLogic
from .parallel_analysts import run_analysts_parallel


# Per-node state fields recorded as that agent's INPUT summary. The point of
# the agent monitor is verifying that each agent actually received its
# upstream content, so paths mirror what each node's prompt consumes.
_NODE_INPUT_PATHS = {
    "Researcher Round": [
        "market_report", "sentiment_report", "news_report", "fundamentals_report",
        "investment_debate_state.judge_critique_bull",
        "investment_debate_state.judge_critique_bear",
    ],
    "Judge Researcher": [
        "investment_debate_state.bull_history",
        "investment_debate_state.bear_history",
    ],
    "Research Manager": [
        "investment_debate_state.history",
        "market_report", "sentiment_report", "news_report", "fundamentals_report",
    ],
    "Trader": [
        "investment_plan",
        "market_report", "sentiment_report", "news_report", "fundamentals_report",
    ],
    "Aggressive Analyst": ["trader_investment_plan", "risk_debate_state.history"],
    "Conservative Analyst": ["trader_investment_plan", "risk_debate_state.history"],
    "Neutral Analyst": ["trader_investment_plan", "risk_debate_state.history"],
    "Portfolio Manager": [
        "investment_plan", "trader_investment_plan", "risk_debate_state.history",
    ],
}


def _dig(state, path: str):
    cur = state
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _summarize_output(result: dict) -> dict:
    """Extract the human-readable text fields an agent produced."""
    out = {}
    for k, v in result.items():
        if k == "messages":
            continue
        if isinstance(v, str) and v:
            out[k] = v
        elif isinstance(v, dict):
            for kk in (
                "current_response", "judge_decision",
                "judge_critique_bull", "judge_critique_bear",
            ):
                vv = v.get(kk)
                if isinstance(vv, str) and vv:
                    out[f"{k}.{kk}"] = vv
    return out


def _recorded(name: str, fn):
    """Wrap a graph node so its inputs/outputs land in the run record."""
    def wrapper(state):
        rec = get_recorder()
        rec.agent_start(
            name,
            inputs={p: _dig(state, p) for p in _NODE_INPUT_PATHS.get(name, [])},
        )
        result = fn(state)
        rec.agent_finish(name, output=_summarize_output(result))
        return result
    return wrapper


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: Dict[str, ToolNode],
        conditional_logic: ConditionalLogic,
        memory_store=None,
    ):
        """Initialize with required components."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.conditional_logic = conditional_logic
        self.memory_store = memory_store

    def setup_graph(
        self,
        selected_analysts=None,
        parallel: bool = True,
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts: List of analyst types to include. Options:
                ``"market"``, ``"social"``, ``"news"``, ``"fundamentals"``.
            parallel: When True the analysts run concurrently (one thread
                each); when False they run sequentially through the same
                isolated-ReAct code path (max_workers=1) — for local models.
        """
        if selected_analysts is None:
            selected_analysts = ["market", "social", "news", "fundamentals"]

        if len(selected_analysts) == 0:
            raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")

        # ------------------------------------------------------------------
        # Build analyst components
        # ------------------------------------------------------------------
        analyst_nodes: Dict[str, Any] = {}
        tool_nodes: Dict[str, Any] = {}

        if "market" in selected_analysts:
            analyst_nodes["market"] = create_market_analyst(self.quick_thinking_llm)
            tool_nodes["market"] = self.tool_nodes["market"]

        if "social" in selected_analysts:
            analyst_nodes["social"] = create_social_media_analyst(
                self.quick_thinking_llm,
                self.memory_store,
            )
            tool_nodes["social"] = self.tool_nodes["social"]

        if "news" in selected_analysts:
            analyst_nodes["news"] = create_news_analyst(self.quick_thinking_llm)
            tool_nodes["news"] = self.tool_nodes["news"]

        if "fundamentals" in selected_analysts:
            analyst_nodes["fundamentals"] = create_fundamentals_analyst(
                self.quick_thinking_llm
            )
            tool_nodes["fundamentals"] = self.tool_nodes["fundamentals"]

        # ------------------------------------------------------------------
        # Build researcher / manager / trader nodes
        # ------------------------------------------------------------------
        researcher_round_node = create_researcher_round(
            self.quick_thinking_llm, self.memory_store
        )
        judge_researcher_node = create_judge_researcher(
            self.deep_thinking_llm,
            fallback_on_failure=False,
        )
        research_manager_node = create_research_manager(
            self.deep_thinking_llm, self.memory_store
        )
        trader_node = create_trader(self.quick_thinking_llm, self.memory_store)

        # ------------------------------------------------------------------
        # Build risk-analysis nodes
        # ------------------------------------------------------------------
        aggressive_analyst = create_aggressive_debator(self.quick_thinking_llm)
        neutral_analyst = create_neutral_debator(self.quick_thinking_llm)
        conservative_analyst = create_conservative_debator(self.quick_thinking_llm)
        portfolio_manager_node = create_portfolio_manager(
            self.deep_thinking_llm, self.memory_store
        )

        # ------------------------------------------------------------------
        # Construct the workflow
        # ------------------------------------------------------------------
        workflow = StateGraph(AgentState)

        # Nodes shared by both modes (wrapped so the run record captures
        # each agent's received inputs and produced outputs)
        workflow.add_node("Researcher Round", _recorded("Researcher Round", researcher_round_node))
        workflow.add_node("Judge Researcher", _recorded("Judge Researcher", judge_researcher_node))
        workflow.add_node("Research Manager", _recorded("Research Manager", research_manager_node))
        workflow.add_node("Trader", _recorded("Trader", trader_node))
        workflow.add_node("Portfolio Manager", _recorded("Portfolio Manager", portfolio_manager_node))

        # ------------------------------------------------------------------
        # Analyst layer — single execution path; `parallel` only controls
        # the worker count (4 threads for cloud, 1 for local models).
        # ------------------------------------------------------------------
        _selected = list(selected_analysts)
        _analyst_nodes = dict(analyst_nodes)
        _tool_nodes = dict(tool_nodes)
        _workers = len(_selected) if parallel else 1

        def _run_analysts_node(state, *, _sel=_selected, _an=_analyst_nodes,
                               _tn=_tool_nodes, _w=_workers):
            configs = [(_t, _an[_t], _tn[_t]) for _t in _sel]
            state_context = {
                "company_of_interest": state["company_of_interest"],
                "trade_date": state["trade_date"],
            }
            # Each analyst runs its full ReAct loop with an isolated message
            # history. Only report fields are returned; shared messages stay
            # intact, so no msg-clear nodes are needed.
            return run_analysts_parallel(
                configs, state["messages"], state_context, max_workers=_w
            )

        workflow.add_node("Run Analysts", _run_analysts_node)
        workflow.add_edge(START, "Run Analysts")
        workflow.add_edge("Run Analysts", "Researcher Round")

        # ------------------------------------------------------------------
        # Researcher / Judge debate edges (same for both analyst modes)
        #
        # Flow:  Researcher Round ──→ Judge Researcher (if judge_count < judge_iterations)
        #                         └──→ Research Manager (otherwise)
        #        Judge Researcher ──→ Researcher Round  (always — static edge)
        # ------------------------------------------------------------------
        workflow.add_conditional_edges(
            "Researcher Round",
            self.conditional_logic.should_continue_to_judge,
            {
                "Judge Researcher": "Judge Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_edge("Judge Researcher", "Researcher Round")
        workflow.add_edge("Research Manager", "Trader")

        # ------------------------------------------------------------------
        # Risk-analyst layer — always sequential (preserves within-round
        # immediate rebuttal: Aggressive → Conservative → Neutral → …)
        # ------------------------------------------------------------------
        workflow.add_node("Aggressive Analyst", _recorded("Aggressive Analyst", aggressive_analyst))
        workflow.add_node("Neutral Analyst", _recorded("Neutral Analyst", neutral_analyst))
        workflow.add_node("Conservative Analyst", _recorded("Conservative Analyst", conservative_analyst))

        workflow.add_edge("Trader", "Aggressive Analyst")
        workflow.add_conditional_edges(
            "Aggressive Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Conservative Analyst": "Conservative Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Conservative Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Neutral Analyst": "Neutral Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Aggressive Analyst": "Aggressive Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )

        workflow.add_edge("Portfolio Manager", END)

        return workflow.compile()
