"""Graph state 定义.

规格要求必须是这些确切字段:

    messages        — 对话历史;通过 add_messages reducer 追加
    retrieved_docs  — 本轮由检索工具取回的文本块
    current_query   — 驱动本轮的(可能被改写过的)查询
    retry_count     — 已消耗的 rewrite 重试次数
    final_answer    — generate_node 产出的回答(或 None)
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class KnowledgeState(TypedDict, total=False):
    """所有 graph node 共享的 state."""

    messages: Annotated[list[AnyMessage], add_messages]
    retrieved_docs: list[dict[str, Any]]
    current_query: str
    retry_count: int
    final_answer: str | None
