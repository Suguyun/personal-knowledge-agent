"""Graph state definition.

The spec mandates these exact fields:

    messages        — conversation history; appended via add_messages reducer
    retrieved_docs  — chunks fetched by the search tool for the current turn
    current_query   — the (possibly rewritten) query driving this turn
    retry_count     — how many rewrite-retries have been consumed
    final_answer    — the answer produced by generate_node (or None)
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class KnowledgeState(TypedDict, total=False):
    """State shared across all graph nodes."""

    messages: Annotated[list[AnyMessage], add_messages]
    retrieved_docs: list[dict[str, Any]]
    current_query: str
    retry_count: int
    final_answer: str | None
