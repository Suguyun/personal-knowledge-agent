"""Graph assembly: StateGraph + nodes + edges + SQLite checkpointer.

Flow (per spec):

    START → intent_router → (knowledge_search_node | direct_response_node)
                          → rerank_node → generate_node → END
        generate_node ──quality ok──→ END
        generate_node ──bad + retry budget──→ rewrite_query_node → knowledge_search_node

The checkpointer persists thread state to SQLite (langgraph-checkpoint-sqlite),
keyed by `thread_id`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from config import Settings, get_settings
from graph import nodes
from graph.edges import (
    BRANCH_DIRECT,
    BRANCH_KNOWLEDGE,
    route_after_generate,
    route_after_rewrite,
    route_intent,
)
from graph.llm import ZhipuLLM
from graph.state import KnowledgeState
from rag.retriever import Retriever
from rag.vectorstore import VectorStore
from tools import get_tools

logger = logging.getLogger(__name__)


async def build_graph(
    settings: Settings | None = None,
    retriever: Retriever | None = None,
    store: VectorStore | None = None,
    llm: Any | None = None,
    client: Any | None = None,
    stream_tokens=None,
):
    """Assemble and compile the agent graph.

    Async because `AsyncSqliteSaver` must be constructed inside a running
    event loop. Await it from any async context, or use `build_graph_sync`
    from a plain script.

    Args:
        settings: App settings (defaults to global).
        retriever: Pre-built Retriever (built from store if omitted).
        store: Pre-built VectorStore (built from defaults if omitted).
        llm: ZhipuLLM adapter over zai-sdk (built if omitted).
        client: Kept for backward compatibility; no longer used internally.
        stream_tokens: Optional `callable(str)` called with each streamed token
                       during generation (used by the CLI).

    Returns:
        A compiled `CompiledStateGraph` ready for `ainvoke`/`astream` with
        `config={"configurable": {"thread_id": ...}}`.
    """
    settings = settings or get_settings()
    settings.ensure_dirs()

    # --- Dependency wiring ------------------------------------------------
    store = store or VectorStore(settings)
    retriever = retriever or Retriever(settings, store=store)

    tools = get_tools(settings=settings, retriever=retriever, store=store)

    # The official zai-sdk ZhipuAiClient is used for every LLM call. Thinking
    # stays enabled by default (GLM-5.2's own strength); `stream_tokens` is
    # forwarded so the interactive CLI can stream tokens when it wires a
    # callback. max_tokens is generous so the reasoning trace never eats the
    # final answer's budget.
    llm = llm or (
        ZhipuLLM(
            api_key=settings.zhipu_api_key,
            model=settings.resolve_model_name,
            base_url=settings.openai_base_url,
            temperature=0.3,
            stream_tokens=stream_tokens,
        )
        if settings.zhipu_api_key
        else None
    )

    nodes._set_deps(
        settings=settings,
        llm=llm,
        tools=tools,
        client=None,  # legacy slot, unused with ZhipuLLM
        stream_tokens=stream_tokens,
    )

    # --- Graph wiring ------------------------------------------------------
    graph = StateGraph(KnowledgeState)

    graph.add_node("intent_router", nodes.intent_router)
    graph.add_node("knowledge_search_node", nodes.knowledge_search)
    graph.add_node("direct_response_node", nodes.direct_response)
    graph.add_node("rerank_node", nodes.rerank_node)
    graph.add_node("generate_node", nodes.generate_node)
    graph.add_node("rewrite_query_node", nodes.rewrite_query_node)

    graph.add_edge(START, "intent_router")

    graph.add_conditional_edges(
        "intent_router",
        route_intent,
        {
            BRANCH_KNOWLEDGE: "knowledge_search_node",
            BRANCH_DIRECT: "direct_response_node",
        },
    )

    graph.add_edge("knowledge_search_node", "rerank_node")
    graph.add_edge("rerank_node", "generate_node")
    graph.add_edge("direct_response_node", END)

    graph.add_conditional_edges(
        "generate_node",
        route_after_generate,
        {
            "rewrite": "rewrite_query_node",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "rewrite_query_node",
        route_after_rewrite,
        {"knowledge_search": "knowledge_search_node"},
    )

    # --- Checkpointer -------------------------------------------------------
    # All graph nodes are async, so the checkpointer must be the async sqlite
    # saver (`SqliteSaver` raises NotImplementedError under ainvoke).
    # `AsyncSqliteSaver.from_conn_string` is an async context manager that
    # closes the connection on exit — which would break checkpointing for the
    # life of the graph. We open the aiosqlite connection ourselves and keep
    # it open so the compiled graph can checkpoint across every ainvoke.
    import aiosqlite

    conn = aiosqlite.connect(str(settings.sqlite_checkpoint_path))
    saver = AsyncSqliteSaver(conn)
    compiled = graph.compile(checkpointer=saver)

    # Keep the connection reachable so short-lived scripts can close it
    # gracefully (see `close_graph`) instead of leaking a thread on exit.
    setattr(compiled, "_checkpointer_conn", conn)
    logger.info("Compiled knowledge agent graph (checkpointer=%s).",
                settings.sqlite_checkpoint_path)
    return compiled


async def run_agent(
    query: str,
    thread_id: str = "default",
    settings: Settings | None = None,
    graph=None,
    **kwargs,
) -> str:
    """Convenience async entry point: ask one query, return the final answer.

    Args:
        query: The user message.
        thread_id: Checkpoint thread id (conversation continuity).
        settings: App settings.
        graph: Pre-built compiled graph (built fresh if omitted).
        **kwargs: Forwarded to build_graph when graph is omitted.

    Returns:
        The agent's final answer text.
    """
    settings = settings or get_settings()
    graph = graph or await build_graph(settings=settings, **kwargs)

    from langchain_core.messages import HumanMessage

    # The add_messages reducer appends to existing checkpointed history for
    # this thread, giving multi-turn continuity.
    state: KnowledgeState = {"messages": [HumanMessage(content=query)], "retry_count": 0}
    result = await graph.ainvoke(
        state,
        config={"configurable": {"thread_id": thread_id}},
    )
    answer = result.get("final_answer") or ""
    return answer


def build_graph_sync(*args, **kwargs):
    """Sync convenience wrapper for `build_graph` (runs its own event loop).

    Note: the returned compiled graph's checkpointer is bound to the loop that
    was used to build it, so in-process reuse from a different loop is unsafe.
    Prefer `await build_graph(...)` inside your app's own event loop.
    """
    return asyncio.run(build_graph(*args, **kwargs))


async def close_graph(graph) -> None:
    """Gracefully close a graph's checkpoint connection (idempotent).

    Call this before your event loop shuts down when using `asyncio.run`
    style short-lived scripts, so the aiosqlite worker thread doesn't try to
    write to a closed loop.
    """
    conn = getattr(graph, "_checkpointer_conn", None)
    if conn is not None:
        try:
            await conn.close()
        except Exception:
            pass  # already closed or loop shutting down


async def clear_thread(graph, thread_id: str) -> None:
    """Delete every checkpoint stored for `thread_id` (best-effort).

    One-shot runs mint a unique thread id so they cannot inherit history from
    an earlier run — but that id is never reused, so without this the
    checkpoint DB would grow by one abandoned thread per invocation (the DB is
    never pruned automatically). Clearing it keeps the isolation without the
    leak. Idempotent: a missing thread deletes nothing.
    """
    saver = getattr(graph, "checkpointer", None)
    deleter = getattr(saver, "adelete_thread", None)
    if deleter is None:
        logger.debug("Checkpointer has no adelete_thread; skipping cleanup.")
        return
    try:
        await deleter(thread_id)
    except Exception:
        logger.warning("Could not clear checkpoints for thread %s",
                       thread_id, exc_info=True)


__all__ = [
    "build_graph",
    "build_graph_sync",
    "run_agent",
    "close_graph",
    "clear_thread",
]
