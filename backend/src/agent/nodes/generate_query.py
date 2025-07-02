import os
import uuid

from langchain_core.prompts.prompt import PromptTemplate

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

def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
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
    memory_tools = get_memory_tools(user_id)

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


    # Format the prompt
    current_date = get_current_date()
    # formatted_prompt = query_writer_instructions.format(
    #     current_date=current_date,
    #     research_topic=get_research_topic(state["messages"]),
    #     number_queries=state["initial_search_query_count"],
    # )
    #
    prompt_variables = {
        "current_date": current_date,
        "research_topic": get_research_topic(state["messages"]),
        "number_queries": state["initial_search_query_count"],
    }

    react_agent = create_react_agent(
        model=llm,
        tools=memory_tools,
        prompt=PromptTemplate.from_template(query_writer_instructions, partial_variables=prompt_variables),
        response_format=SearchQueryList
    )
    # Generate the search queries
    # result = structured_llm.invoke(formatted_prompt)
    result = react_agent.invoke(prompt_variables)
    print(result)
    return {"search_query": parse_json_from_response(result['messages'][-1].content)['query']}
