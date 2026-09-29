"""Graph 组装:StateGraph + nodes + edges + SQLite checkpointer.

流程(按规格):

    START → intent_router → (knowledge_search_node | direct_response_node)
                          → rerank_node → generate_node → END
        generate_node ──质量达标──→ END
        generate_node ──不达标 + 还有 retry 预算──→ rewrite_query_node → knowledge_search_node

checkpointer 把 thread state 持久化到 SQLite(langgraph-checkpoint-sqlite),
以 `thread_id` 为键.
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
from graph.llm import OpenAICompatLLM
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
    """组装并编译 agent graph.

    之所以是 async,是因为 `AsyncSqliteSaver` 必须在运行中的事件循环里构造.
    可在任意 async 上下文中 await 它,或在普通脚本里用 `build_graph_sync`.

    Args:
        settings: 应用配置(默认为全局).
        retriever: 预先构建的 Retriever(省略时基于 store 构建).
        store: 预先构建的 VectorStore(省略时用默认值构建).
        llm: OpenAI 兼容的 LLM 适配器(省略时按 settings 构建).
        client: 为向后兼容保留;内部已不再使用.
        stream_tokens: 可选 `callable(str)`,在生成过程中每收到一个流式
                       token 即调用(供 CLI 使用).

    Returns:
        一个已编译的 `CompiledStateGraph`,可直接配合
        `config={"configurable": {"thread_id": ...}}` 用于 `ainvoke`/`astream`.
    """
    settings = settings or get_settings()
    settings.ensure_dirs()

    # --- 依赖装配 ----------------------------------------------------------
    store = store or VectorStore(settings)
    retriever = retriever or Retriever(settings, store=store)

    tools = get_tools(settings=settings, retriever=retriever, store=store)

    # 用哪家模型完全由 settings 决定(llm_base_url / llm_model / llm_api_key).
    # Thinking 保持模型默认(DeepSeek 与 GLM 都默认开启);转发
    # `stream_tokens`,以便交互式 CLI 接上回调后能流式输出 token.max_tokens
    # 给得宽裕,避免推理轨迹吃掉最终回答的预算.
    llm = llm or (
        OpenAICompatLLM(
            api_key=settings.llm_api_key,
            model=settings.resolve_model_name,
            base_url=settings.llm_base_url,
            temperature=0.3,
            stream_tokens=stream_tokens,
        )
        if settings.llm_api_key
        else None
    )

    nodes._set_deps(
        settings=settings,
        llm=llm,
        tools=tools,
        client=None,  # 遗留占位,OpenAICompatLLM 下未使用
        stream_tokens=stream_tokens,
    )

    # --- Graph 接线 --------------------------------------------------------
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
    # 所有 graph node 都是 async,因此 checkpointer 必须是 async 版 sqlite
    # saver(`SqliteSaver` 在 ainvoke 下会抛 NotImplementedError).
    # `AsyncSqliteSaver.from_conn_string` 是个 async 上下文管理器,退出时会关
    # 闭连接 —— 那会破坏整个 graph 生命周期的 checkpoint.我们自己打开
    # aiosqlite 连接并保持打开,这样编译后的 graph 能在每次 ainvoke 之间做
    # checkpoint.
    import aiosqlite

    conn = aiosqlite.connect(str(settings.sqlite_checkpoint_path))
    saver = AsyncSqliteSaver(conn)
    compiled = graph.compile(checkpointer=saver)

    # 让连接保持可达,好让短生命周期脚本能优雅关闭它(见 `close_graph`),
    # 而不是在退出时泄漏一个线程.
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
    """便捷的 async 入口:问一个查询,返回最终回答.

    Args:
        query: 用户消息.
        thread_id: checkpoint thread id(保证对话连续性).
        settings: 应用配置.
        graph: 预先构建好的已编译 graph(省略时新构建).
        **kwargs: graph 省略时转发给 build_graph.

    Returns:
        agent 的最终回答文本.
    """
    settings = settings or get_settings()
    graph = graph or await build_graph(settings=settings, **kwargs)

    from langchain_core.messages import HumanMessage

    # add_messages reducer 会把消息追加到该 thread 已有的 checkpoint 历史之后,
    # 从而实现多轮连续性.
    state: KnowledgeState = {"messages": [HumanMessage(content=query)], "retry_count": 0}
    result = await graph.ainvoke(
        state,
        config={"configurable": {"thread_id": thread_id}},
    )
    answer = result.get("final_answer") or ""
    return answer


def build_graph_sync(*args, **kwargs):
    """`build_graph` 的同步便捷包装(自行运行事件循环).

    注意:返回的已编译 graph 的 checkpointer 绑定在构建它的那个事件循环上,
    因此从另一个循环在进程内复用它是不安全的.推荐在你自己应用的事件循环里
    `await build_graph(...)`.
    """
    return asyncio.run(build_graph(*args, **kwargs))


async def close_graph(graph) -> None:
    """优雅关闭 graph 的 checkpoint 连接(幂等).

    在使用 `asyncio.run` 风格的短生命周期脚本时,请在事件循环关闭前调用它,
    以免 aiosqlite 工作线程试图往已关闭的循环里写.
    """
    conn = getattr(graph, "_checkpointer_conn", None)
    if conn is not None:
        try:
            await conn.close()
        except Exception:
            pass  # 已关闭,或事件循环正在退出


async def clear_thread(graph, thread_id: str) -> None:
    """删除为 `thread_id` 存储的所有 checkpoint(尽力而为).

    一次性运行会生成唯一 thread id,因此不会继承早先运行的历史 —— 但那个 id
    永不复用,所以若不清理,checkpoint DB 会随每次调用增长一个被遗弃的
    thread(该 DB 从不自动裁剪).清掉它既能保持隔离又不会泄漏.幂等:
    thread 不存在时什么也不删.
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
