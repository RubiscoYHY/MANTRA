
from tradingagents.agents.utils.agent_utils import build_instrument_context


def create_research_manager(llm, memory_store):
    def research_manager_node(state) -> dict:
        ticker = state["company_of_interest"]
        instrument_context = build_instrument_context(ticker)
        history = state["investment_debate_state"].get("history", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        investment_debate_state = state["investment_debate_state"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory_store.retrieve_reflections(
            ticker=ticker, role="invest_judge", query=curr_situation, n_results=2
        )

        past_memory_str = ""
        for hit in past_memories:
            past_memory_str += hit["text"] + "\n\n"

        prompt = f"""You are the Research Manager. Your role is to synthesize the outcome of a Judge-mediated investment debate and produce a definitive investment recommendation. You are the final decision-maker of the research phase. You do not facilitate the debate — that was the Judge's role — and you are not the Portfolio Manager, who makes the final trading decision downstream.

GROUNDING RULE:
Your analysis and recommendation MUST be based solely on:
1. Arguments and evidence that appeared in the debate history below.
2. Your past memories of similar situations.
You may NOT introduce new facts, data, or analysis that did not appear in the debate. If the debate does not provide sufficient evidence to support a point, do not assert it.

HOW TO READ THE DEBATE HISTORY:
The history contains not just the analysts' arguments but also the Judge's critiques and each analyst's subsequent responses. This is your primary quality signal. Work through it in four steps before forming your recommendation.

Step 1 — Identify and discard retracted claims:
Look for cases where the Judge asked an analyst to source a specific claim and the analyst acknowledged it lacked direct support, retracted it, or significantly revised it. Treat these claims as no longer part of that analyst's case. Do not count them in your evaluation.

Step 2 — Evaluate contested interpretations:
Where the Judge identified that both analysts interpreted the same fact or phenomenon differently and asked each side to justify their reading, assess the quality of each side's subsequent response. A response grounded in specific report evidence carries more weight than one that is vague, evasive, or merely reasserts the original claim without new support.

Step 3 — Identify genuine unresolved conflicts:
If the Judge raised a cross-examination point and both sides provided substantive, report-grounded responses that still reach opposite conclusions, treat this as a genuine uncertainty. Do not resolve it by fiat in favor of either side. Acknowledge it explicitly in your reasoning and reflect it in the confidence level of your recommendation.

Step 4 — Synthesize from surviving arguments:
Based only on arguments that survived the full Judge process — not retracted, not refuted by a clearly superior counter-response — determine which side has the stronger overall evidentiary and logical case. Your recommendation must follow from this assessment.

PAST EXPERIENCE:
Take into account your reflections from similar past situations. Use these to refine your judgment, especially where current evidence is ambiguous.

OUTPUT STRUCTURE — use the following three sections:

Recommendation: State Buy, Sell, or Hold. Avoid Hold unless the surviving arguments from both sides are genuinely balanced after applying the four steps above; do not use Hold as a default when the analysis is difficult.

Reasoning: Explain which surviving arguments drove your conclusion and why. For each unresolved conflict identified in Step 3, state explicitly why it does or does not change your recommendation.

Investment Plan for the Trader: Concrete, actionable guidance — key price levels or conditions to watch, position sizing considerations, and primary risk factors that could invalidate the thesis.

---

Past reflections on similar situations:
\"{past_memory_str}\"

{instrument_context}

Debate History:
{history}"""
        response = llm.invoke(prompt)

        new_investment_debate_state = {
            "judge_decision": response.content,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": response.content,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": response.content,
        }

    return research_manager_node
