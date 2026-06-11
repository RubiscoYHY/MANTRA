from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.agents.utils.memory_store import build_situation_digest


def create_trader(llm, memory_store):
    def trader_node(state):
        company_name = state["company_of_interest"]
        instrument_context = build_instrument_context(company_name)
        investment_plan = state["investment_plan"]
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
            ticker=company_name, role="trader", query=situation_digest, n_results=2
        )
        past_memory_str = (
            "\n\n".join(hit["text"] for hit in past_memories)
            if past_memories else "No sufficiently similar past situation found."
        )

        context = {
            "role": "user",
            "content": f"Based on a comprehensive analysis by a team of analysts, here is an investment plan tailored for {company_name}. {instrument_context} This plan incorporates insights from current technical market trends, macroeconomic indicators, and social media sentiment. Use this plan as a foundation for evaluating your next trading decision.\n\nProposed Investment Plan: {investment_plan}\n\nLeverage these insights to make an informed and strategic decision.",
        }

        messages = [
            {
                "role": "system",
                "content": f"""You are a trading agent analyzing market data to make investment decisions. Based on your analysis, provide a specific recommendation to buy, sell, or hold. End with a firm decision and always conclude your response with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**' to confirm your recommendation. Apply lessons from past decisions to strengthen your analysis. Here are reflections from similar situations you traded in and the lessons learned: {past_memory_str}""",
            },
            context,
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
        }

    return trader_node
