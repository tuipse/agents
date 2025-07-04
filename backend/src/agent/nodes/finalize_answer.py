import os

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from src.agent.state import (
    OverallState,
)
from src.agent.configuration import Configuration
from src.agent.prompts import (
    get_current_date,
    answer_instructions,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from src.agent.utils import (
    get_research_topic,
)
from src.agent.memory.tools import get_memory_tools
from src.agent.memory.tools import search_in_memory


def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model

    user_id = "0" if state.get("user_id") is None else state.get("user_id")
    memory_items = search_in_memory('', user_id,  "long-term-memory")

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
        memory=memory_items
    )


    # init Reasoning Model, default to Gemini 2.5 Flash
    llm = ChatGoogleGenerativeAI(
        model=reasoning_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    result = llm.invoke(formatted_prompt)

    # Replace the short urls with the original urls and add all used urls to the sources_gathered
    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in result.content:
            result.content = result.content.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }
