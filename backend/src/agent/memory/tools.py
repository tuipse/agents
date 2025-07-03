import os
import uuid
from typing import Any
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.config import get_store


def get_memory_tools(lang_graph_user_id):
    manage_memory_tool = create_manage_memory_tool(
        namespace=("email_assistant", lang_graph_user_id, "collection")
    )
    search_memory_tool = create_search_memory_tool(
        namespace=("email_assistant", lang_graph_user_id, "collection")
    )

    return [manage_memory_tool, search_memory_tool]

def add_to_memory(key:str, content: dict[str, Any], lang_graph_user_id: str, namespace: str = "memory"):
    store = get_store()
    store.put((namespace, lang_graph_user_id), key, content)

def remove_from_memory(key: str, lang_graph_user_id: str, namespace: str = "memory"):
    store = get_store()
    store.delete((namespace, lang_graph_user_id), key)

def search_in_memory(query: str, lang_graph_user_id: str, namespace: str = "memory"):
    store = get_store()
    results_of_search = store.search((namespace, lang_graph_user_id), limit=999999)
    memory_items = []
    for item in results_of_search:
        memory_items.append(' - Memory (' + item.key + '): ' + item.value['content'])

    return '\n'.join(memory_items)
