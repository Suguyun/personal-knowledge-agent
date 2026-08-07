"""Tool registry for the knowledge agent.

The three tools are runtime-injected with their dependencies (retriever,
vector store, settings). `get_tools()` returns the `@tool`-decorated callables
that are bound to `bind_tools` and passed to the LLM in native OpenAI function
calling format.

Usage:
    tools = get_tools(retriever=retriever, store=store)
    tool_map = {t.name: t for t in tools}   # used by the node layer
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from config import Settings, get_settings

# The canonical order in which tools are presented to the LLM.
from .create_note import _make_create_note
from .knowledge_search import _make_knowledge_search
from .list_documents import _make_list_documents


def get_tools(
    settings: Settings | None = None,
    retriever=None,
    store=None,
) -> list[BaseTool]:
    """Build the tool list with runtime dependencies injected.

    Args:
        settings: App settings (defaults to global).
        retriever: A `rag.Retriever` instance used by `knowledge_search`.
        store: A `rag.VectorStore` instance used by `list_documents` and
               `create_note`.

    Returns:
        List of langchain BaseTool instances, ready for `bind_tools`.
    """
    settings = settings or get_settings()
    return [
        _make_knowledge_search(retriever=retriever),
        _make_list_documents(store=store),
        _make_create_note(store=store, notes_dir=settings.notes_dir),
    ]


__all__ = ["get_tools"]
