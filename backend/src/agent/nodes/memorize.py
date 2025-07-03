import os
import uuid
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
from langchain.load import dumps
from src.agent.utils import parse_json_from_response
from src.agent.memory.tools import add_to_memory, search_in_memory


memory_instructions = """
Analyze the following conversation state, list all facts that are not listed already in your memory and entities mentioned in order for memorize them.

Return them in a JSON array of facts, example:
```json
[
"His name is Gabriel",
"The user researched about war in 2025",
"The user is interested in the history of the Roman Empire"
...
]
```

Important: do not duplicate facts.
"""
memory_instructions_content = """
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

    # init Gemini 2.0 Flash
    llm = ChatGoogleGenerativeAI(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    # Format the prompt
    current_date = get_current_date()

    prompt_variables = {
        "current_date": current_date,
        "state": dumps(state, pretty=True),
    }

    response = llm.invoke(input=[
        SystemMessage(content=memory_instructions),
        memory_instructions_content.format(**prompt_variables),
    ], config=tool_config)


    resulted_json = parse_json_from_response(response.content)

    # Check if resulted_json is an array
    if isinstance(resulted_json, list):
        for item in resulted_json:
            similar_memories = search_in_memory(item, user_id, "long-term-memory")
            # Check if its a list and has less than 4 records
            if isinstance(similar_memories, list) and len(similar_memories) < 4:
                add_to_memory(uuid.uuid4().hex, {'content': item}, user_id, "long-term-memory")
            else:
                print(f"Skipping item '{item}' as it already has more than 4 similar memories.")
    else:
        raise ValueError("resulted_json is not an array")

    return {}
