"""`knowledge_search` tool — the primary knowledge base access point.

The docstring is the tool schema the model sees. Because GLM-5.2 relies
heavily on docstrings for tool understanding, the description is written in
Chinese (matching the system prompt) with the parameters spelled out
explicitly. The implementation delegates to the injected `Retriever`.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _make_knowledge_search(retriever: Any | None):
    """Closure builder so the tool is bound to its retriever at runtime."""

    @tool
    def knowledge_search(query: str, filters: Optional[dict[str, Any]] = None) -> str:
        """在你的个人知识库中检索与查询最相关的信息片段，并返回其原文与出处元数据。

        知识库由本地 markdown 文档构建（含全部笔记、学习资料、项目文档等）。
        此工具是回答知识类问题的唯一入口，回答任何与个人知识相关的问题时都必须优先调用。

        Args:
            query: 需要检索的自然语言查询语句，应完整表达用户的真实意图。查询可以
                   包含中文或英文，例如“如何配置 GLM-5.2 的函数调用”、“2026年OKR”等。
            filters: 可选的元数据过滤条件，键值对形式。常用键包括：
                   - "source_doc": 限定只检索某个文档，例如 {"source_doc": "meeting-2026.md"}
                   - "section_header": 限定某个章节
                   不需要过滤时请传空对象或省略。

        Returns:
            以 JSON 字符串返回最多 5 条最相关的检索结果。每条结果包含：
              - content: 知识片段原文（可能为 markdown 格式）
              - metadata: 来源元数据，其中 source_doc 为来源文档名，
                section_header 为章节名
              - score: 相关度得分（0-1，越大越相关）
            若知识库中没有相关内容，返回空数组。
        """
        if not retriever:
            raise ValueError("knowledge_search 未绑定 retriever，无法执行检索。")
        try:
            hits = retriever.search(query=query, filters=filters or None)
        except Exception as exc:  # structured error → LLM decides next step
            logger.exception("knowledge_search failed")
            return f"检索失败: {exc}"
        return _format_hits(hits)

    return knowledge_search


def _format_hits(hits: list[dict[str, Any]]) -> str:
    """Serialize retrieval results as a compact JSON string."""
    import json

    payload = []
    for hit in hits:
        meta = hit.get("metadata") or {}
        payload.append(
            {
                "content": hit.get("content", ""),
                "metadata": {
                    "source_doc": meta.get("source_doc", ""),
                    "section_header": meta.get("section_header", ""),
                    "chunk_index": meta.get("chunk_index"),
                },
                "score": round(float(hit.get("score", 0.0)), 4),
            }
        )
    return json.dumps(payload, ensure_ascii=False)


__all__ = ["_make_knowledge_search"]
