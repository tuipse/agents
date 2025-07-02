import os
from langchain_core.messages.system import SystemMessage
from langchain_core.runnables import RunnableConfig

from src.agent.state import (
    OverallState,
    QueryGenerationState,
)
from src.agent.configuration import Configuration
from src.agent.prompts import (
    get_current_date,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from src.agent.memory.tools import get_memory_tools
from langchain.load import dumps


memory_instructions = """
Use the `manage_memory` function to save all the information about the following conversation state:
```json
{state}
```
"""

def memorize(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
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
    # 2) build a new config that specifies the tool you want downstream
    tool_config = RunnableConfig(
        metadata={ "tool_to_call": "manage_memory" },
        tags=[ "tool:manage_memory" ]
    )

    user_id = "0" if state.get("user_id") is None else state.get("user_id")
    memory_tools = get_memory_tools(user_id)

    # init Gemini 2.0 Flash
    llm = ChatGoogleGenerativeAI(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    ).bind_tools(memory_tools)

    # Format the prompt
    current_date = get_current_date()

    prompt_variables = {
        "current_date": current_date,
        "state": dumps(state, pretty=True),
    }

    llm.invoke(input=[
        SystemMessage(content=memory_instructions.format(**prompt_variables)),
        memory_instructions.format(**prompt_variables),
    ], config=tool_config)

    return {}
