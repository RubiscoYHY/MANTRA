from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
    get_news,
)
from tradingagents.agents.utils.social_data_tools import (
    get_social_posts_cached,
    finbert_aggregate,
)
from tradingagents.dataflows.config import get_config


def create_social_media_analyst(llm, memory_store=None):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        instrument_context = build_instrument_context(ticker)

        # Pre-fetch and aggregate social media data with FinBERT.
        # get_social_posts_cached() is idempotent within one propagate() call —
        # the analyst node may be re-entered multiple times (ReAct loop) but
        # the network requests are made only once.
        posts = get_social_posts_cached(ticker)
        sentiment_summary = finbert_aggregate(posts)

        # Retrieve historical similar sentiment from memory (causal isolation guaranteed internally)
        historical = []
        sector = []
        if memory_store is not None:
            historical = memory_store.retrieve_similar_sentiment(
                ticker=ticker, query=sentiment_summary, n_results=3
            )
            sector = memory_store.retrieve_sector_sentiment(
                related_tickers=[], query=sentiment_summary, n_results=2
            )

        memory_context = ""
        if historical:
            memory_context += "\n\n--- Historical Similar Sentiment (past analysis) ---\n"
            for entry in historical:
                memory_context += f"- {entry}\n"
            memory_context += "--- End of Historical Sentiment ---"
        if sector:
            memory_context += "\n\n--- Sector Peer Sentiment ---\n"
            for entry in sector:
                memory_context += f"- {entry}\n"
            memory_context += "--- End of Sector Sentiment ---"

        tools = [get_news]

        system_message = (
            "You are a financial social media and sentiment analyst.\n\n"
            "TASK\n"
            "1. Review the FinBERT-processed social media summary below.\n"
            "2. Call the get_news tool to fetch financial news from the past week.\n"
            "3. Write a concise sentiment report covering: overall sentiment trend, "
            "key bullish themes, key bearish themes, notable news catalysts, and "
            "actionable implications for traders.\n"
            "4. End your report with a Markdown table summarising key insights.\n\n"
            "NOISE FILTERING — apply these rules before drawing any conclusion\n"
            "1. Social media feeds are extremely noisy. Many posts only mention the "
            "target ticker in passing while actually discussing a different company, "
            "index, or unrelated topic.\n"
            "2. For every post in the summary, identify its true primary subject. "
            "Discard any post whose primary subject does not match the ticker you are "
            "currently researching.\n"
            "3. If no on-target signal remains after filtering, report that explicitly — "
            "state that no meaningful on-target social sentiment was detected. "
            "Do not fabricate conclusions.\n"
            "4. Never attribute bullish or bearish sentiment from an off-target subject "
            "to the ticker under analysis.\n\n"
            "--- FinBERT Social Media Summary ---\n"
            + sentiment_summary
            + "\n--- End of Social Media Summary ---"
            + memory_context
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    " For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        # Store today's sentiment summary into memory for future retrieval
        if memory_store is not None and report:
            memory_store.store_sentiment_summary(ticker=ticker, summary=report)

        return {
            "messages": [result],
            "sentiment_report": report,
        }

    return social_media_analyst_node
