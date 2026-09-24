"""LangGraph orchestration for the knowledge agent.

Public surface:
    - `KnowledgeState`: the typed graph state
    - `build_graph`:    assemble + compile the StateGraph
    - `run_agent`:      convenience async entry point (single query)
"""

from .builder import (
    build_graph,
    build_graph_sync,
    clear_thread,
    close_graph,
    run_agent,
)
from .state import KnowledgeState

__all__ = [
    "KnowledgeState",
    "build_graph",
    "build_graph_sync",
    "clear_thread",
    "close_graph",
    "run_agent",
]
