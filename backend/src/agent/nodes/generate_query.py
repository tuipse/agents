import os
import uuid

from langchain_core.prompts.prompt import PromptTemplate
from langgraph.types import Send

from src.agent.utils import get_message_intention
from src.agent.tools_and_schemas import SearchQueryList
from langchain_core.runnables import RunnableConfig

from src.agent.state import (
    OverallState,
    QueryGenerationState,
)
from src.agent.configuration import Configuration
from src.agent.prompts import (
    get_current_date,
    query_writer_instructions,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from src.agent.utils import (
    get_research_topic,
)
from src.agent.memory.tools import get_memory_tools
from src.agent.utils import parse_json_from_response
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langgraph.config import get_store

from src.agent.memory.tools import search_in_memory


def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState|Send:
    """LangGraph node that generates search queries based on the User's question.

    Uses Gemini 2.0 Flash to create an optimized search queries for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated queries
    """
    configurable = Configuration.from_runnable_config(config)

    user_id = "0" if state.get("user_id") is None else state.get("user_id")

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # init Gemini 2.0 Flash
    llm = ChatGoogleGenerativeAI(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    memory_items = search_in_memory(state['messages'][-1].content, user_id,  "long-term-memory")

    # Format the prompt
    current_date = get_current_date()

    prompt_variables = {
        "current_date": current_date,
        "research_topic": get_research_topic(state["messages"]),
        "number_queries": state["initial_search_query_count"],
        "memory": memory_items
    }

    message_intention = get_message_intention(state['messages'][-1])

    # Check if list or string in result_search_query is not empty
    if message_intention.intention == 'web_research':
        result = structured_llm.invoke(query_writer_instructions.format(**prompt_variables))
        return Send('web_research', {"search_query": result.query, 'messages': state['messages']})
    else:
        return Send('finalize_answer', state)
