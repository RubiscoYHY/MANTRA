from tradingagents.agents.utils.memory_store import build_situation_digest


def _build_bear_argument(state: dict, llm, memory_store) -> str:
    """
    Core Bear Analyst logic.

    Reads analyst reports and past memories from state, builds a prompt that
    includes the Judge's latest directive (judge_critique_bear) if one exists,
    invokes the LLM, and returns the raw response string (no role prefix).

    Called by researcher_round.py for parallel execution.
    """
    investment_debate_state = state["investment_debate_state"]
    bear_history = investment_debate_state.get("bear_history", "")
    judge_critique = investment_debate_state.get("judge_critique_bear", "")

    ticker = state["company_of_interest"]
    market_research_report = state["market_report"]
    sentiment_report = state["sentiment_report"]
    news_report = state["news_report"]
    fundamentals_report = state["fundamentals_report"]

    # Query with the same compact digest used when reflections were stored,
    # so query and document embeddings live in the same representation space.
    situation_digest = build_situation_digest(
        market_research_report, sentiment_report, news_report, fundamentals_report
    )
    past_memories = memory_store.retrieve_reflections(
        ticker=ticker, role="bear", query=situation_digest, n_results=2
    )
    past_memory_str = (
        "\n\n".join(hit["text"] for hit in past_memories)
        if past_memories else "No sufficiently similar past situation found."
    )

    prompt = f"""You are a Bear Analyst. Your role is to build the strongest possible evidence-based case for why investing in this stock carries unacceptable risk or insufficient reward.

ABSOLUTE RULE — DATA GROUNDING:
Every factual claim, statistic, and market observation in your argument MUST be traceable to one of the four analyst reports provided to you. You may NOT invent data, fabricate trends, or extrapolate beyond what the reports explicitly state. If a fact is not in the reports, do not use it.

YOUR TASK:
Build a logically coherent bearish argument by:
- Identifying risks, structural weaknesses, and negative market signals that are explicitly supported by the reports.
- Constructing a narrative that connects evidence to the risk thesis with clear logical steps.
- Where the evidence permits, challenging bullish assumptions with a credible, evidence-based counter-analysis.

RESPONDING TO JUDGE DIRECTIVES:
If the Judge has issued a directive to you, you MUST address every point it raises:
- If asked to explain or deepen an argument: provide a more detailed, evidence-based analysis of the specific point.
- If asked to identify the source for a claim: cite the specific report and the section or passage.
- If asked to respond to the Bull Analyst's interpretation of a shared phenomenon: provide a substantive counter-analysis grounded in the reports, explaining why your interpretation is more logically consistent with the full body of evidence.

CONDUCT RULES — TWO LEVELS:

Level 1 — Factual claims (point-level, correctable):
If the Judge asks you to source a specific claim and that claim has no direct support in the four reports, you MUST acknowledge that this specific claim lacks direct evidentiary support, withdraw or revise it, and reconstruct that part of your argument using only what the reports do support. Intellectual honesty about individual data points is required. Defending a claim you cannot source is a breach of the data grounding rule.

Level 2 — Directional thesis (protected):
Your overall bearish investment thesis and directional conclusion are your own analytical judgment, formed from the totality of the evidence. You must NOT weaken, qualify, or abandon your directional position in response to Judge directives. You must NOT express that your overall argument is less compelling than you originally presented. Correcting a specific factual claim is entirely compatible with maintaining your investment direction — these are independent. Do NOT thank, compliment, or flatter the Judge. Do NOT express that the Judge has improved your argument or has identified a weakness in your overall case.

Resources available:
Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
Company fundamentals report: {fundamentals_report}
Your previous arguments: {bear_history}
Judge's latest directive to you: {judge_critique if judge_critique else "(none — this is your opening argument)"}
Reflections from similar situations and lessons learned: {past_memory_str}"""

    response = llm.invoke(prompt)
    return response.content
