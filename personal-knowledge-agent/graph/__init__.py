"""知识 agent 的 LangGraph 编排.

对外接口:
    - `KnowledgeState`:带类型的 graph state
    - `build_graph`:组装并编译 StateGraph
    - `run_agent`:便捷的 async 入口(单次查询)
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
