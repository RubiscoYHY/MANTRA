"""UI-agnostic streaming classification and status logic for MANTRA.

This module centralizes the logic shared by the CLI (Rich Live rendering) and the
GUI (Flask SSE queue) for interpreting the LangGraph analysis stream:

- which agent a stream chunk belongs to,
- which report section a chunk updates,
- agent status transitions (pending / in_progress / completed),
- report section ordering and display titles.

It is deliberately free of any UI dependency (no Rich, no Flask, no Queue): every
function operates on plain dicts/lists and either returns a value or mutates a
caller-owned ``agent_status`` mapping in place. Each UI keeps its own rendering
and report-emission mechanics; only the classification/status/section concerns
live here.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Analyst layer constants
# ---------------------------------------------------------------------------

# Ordered list of analysts for status transitions.
ANALYST_ORDER = ["market", "social", "news", "fundamentals"]

# Analyst key -> display agent name.
ANALYST_AGENT_NAMES = {
    "market": "Market Analyst",
    "social": "Social Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
}

# Analyst key -> the report-section key it produces in the graph state.
ANALYST_REPORT_MAP = {
    "market": "market_report",
    "social": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
}


# ---------------------------------------------------------------------------
# Report section constants
# ---------------------------------------------------------------------------

# Report section -> (analyst_key for filtering, finalizing_agent).
#   analyst_key: which analyst selection controls this section (None = always included)
#   finalizing_agent: which agent must be "completed" for this report to count as done
REPORT_SECTIONS = {
    "market_report": ("market", "Market Analyst"),
    "sentiment_report": ("social", "Social Analyst"),
    "news_report": ("news", "News Analyst"),
    "fundamentals_report": ("fundamentals", "Fundamentals Analyst"),
    "investment_plan": (None, "Research Manager"),
    "trader_investment_plan": (None, "Trader"),
    "final_trade_decision": (None, "Portfolio Manager"),
}

# Canonical order of report sections when assembling the full final report.
REPORT_SECTION_ORDER = [
    "market_report",
    "sentiment_report",
    "news_report",
    "fundamentals_report",
    "investment_plan",
    "trader_investment_plan",
    "final_trade_decision",
]

# Human-readable titles for each report section.
REPORT_SECTION_TITLES = {
    "market_report": "Market Analysis",
    "sentiment_report": "Social Sentiment",
    "news_report": "News Analysis",
    "fundamentals_report": "Fundamentals Analysis",
    "investment_plan": "Research Team Decision",
    "trader_investment_plan": "Trading Team Plan",
    "final_trade_decision": "Portfolio Management Decision",
}


# ---------------------------------------------------------------------------
# Initial agent status
# ---------------------------------------------------------------------------

def build_initial_agent_status(selected_analysts, fixed_agents):
    """Build the initial ``agent_status`` mapping (everything "pending").

    Args:
        selected_analysts: iterable of analyst keys (e.g. ["market", "news"]).
        fixed_agents: mapping of team name -> list of agent display names that
            always run (the set differs slightly between UIs).

    Returns:
        dict mapping agent display name -> "pending". The "Media Labeling" row is
        added when the social analyst is selected (it precedes "Social Analyst").
    """
    selected = [a.lower() for a in selected_analysts]
    agent_status: dict[str, str] = {}

    for analyst_key in selected:
        if analyst_key in ANALYST_AGENT_NAMES:
            if analyst_key == "social":
                agent_status["Media Labeling"] = "pending"
            agent_status[ANALYST_AGENT_NAMES[analyst_key]] = "pending"

    for team_agents in fixed_agents.values():
        for agent in team_agents:
            agent_status[agent] = "pending"

    return agent_status


def build_report_section_keys(selected_analysts):
    """Return the report-section keys relevant to the selected analysts.

    Always-included sections (analyst_key is None) are kept; analyst-specific
    sections are only kept when their analyst is selected. Order follows
    ``REPORT_SECTIONS`` insertion order.
    """
    selected = [a.lower() for a in selected_analysts]
    keys = []
    for section, (analyst_key, _) in REPORT_SECTIONS.items():
        if analyst_key is None or analyst_key in selected:
            keys.append(section)
    return keys
