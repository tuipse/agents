import os

from dotenv import load_dotenv
from langgraph.graph import StateGraph
from langgraph.graph import START, END

from src.agent.nodes.memorize import memorize
from src.agent.nodes.continue_to_web_research import continue_to_web_research
from src.agent.nodes.evaluate_research import evaluate_research
from src.agent.nodes.finalize_answer import finalize_answer
from src.agent.nodes.generate_query import generate_query
from src.agent.nodes.reflection import reflection
from src.agent.nodes.web_research import web_research
from src.agent.state import (
    OverallState,
)
from src.agent.configuration import Configuration

load_dotenv()

if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)
builder.add_node("memorize", memorize)

# Set the entrypoint as `generate_query`
# This means that this node is the first one called
builder.add_edge(START, "generate_query")
# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", "memorize")
builder.add_edge("memorize", END)

graph = builder.compile(name="agent")
