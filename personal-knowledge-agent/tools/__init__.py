"""knowledge agent 的 tool 注册表.

三个 tool 在运行时注入各自的依赖(retriever,vector store,settings).
`get_tools()` 返回经 `@tool` 装饰的可调用对象;`build_graph` 把它们交给节点层,
节点层再以原生 OpenAI function-calling tool 的形式暴露给模型,并执行模型请求的
任何调用(见 `nodes._tool_enabled_completion` / `nodes._execute_tool_call`).
`knowledge_search` 还会被 `knowledge_search` graph 节点直接调用,正是它让
retrieve→rerank→generate 这条确定性路径得以工作.

用法:
    tools = get_tools(settings=settings, retriever=retriever, store=store)
    tool_map = {t.name: t for t in tools}   # 节点层使用
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from config import Settings, get_settings

# 向 LLM 展示 tool 时的规范顺序.
from .create_note import _make_create_note
from .knowledge_search import _make_knowledge_search
from .list_documents import _make_list_documents


def get_tools(
    settings: Settings | None = None,
    retriever=None,
    store=None,
) -> list[BaseTool]:
    """构建 tool 列表,并在运行时注入依赖.

    Args:
        settings: 应用 settings(默认为全局 settings).
        retriever: `rag.Retriever` 实例,供 `knowledge_search` 使用.
        store: `rag.VectorStore` 实例,供 `list_documents` 与
               `create_note` 使用.

    Returns:
        langchain BaseTool 实例列表,可直接用于 `bind_tools`.
    """
    settings = settings or get_settings()
    return [
        _make_knowledge_search(retriever=retriever),
        _make_list_documents(store=store),
        _make_create_note(
            store=store,
            notes_dir=settings.notes_dir,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ),
    ]


__all__ = ["get_tools"]
