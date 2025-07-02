from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool

# Memory store
store = InMemoryStore(index={"embed": "openai:text-embedding-3-small"})

def get_memory_tools(lang_graph_user_id):
    manage_memory_tool = create_manage_memory_tool(
        namespace=("email_assistant", lang_graph_user_id, "collection")
    )
    search_memory_tool = create_search_memory_tool(
        namespace=("email_assistant", lang_graph_user_id, "collection")
    )

    return [manage_memory_tool, search_memory_tool]
