def __getattr__(name):
    """Lazy import to prevent circular imports."""
    if name == "graph":
        from agent.graph import graph
        return graph
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ["graph"]
