"""`list_documents` tool — enumerate the knowledge base contents.

Aggregates ChromaDB metadata into a per-document summary (name, tags/sections,
last updated time, chunk count). Used by the model before deciding whether a
query topic exists in the KB, and useful as a "what do I know" overview.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _make_list_documents(store: Any | None):
    """Closure builder so the tool is bound to its vector store at runtime."""

    @tool
    def list_documents() -> str:
        """列出个人知识库中的所有文档及概览信息，无需任何参数。

        返回每个文档的名称、章节标签（由 markdown 标题生成）、最近更新时间和
        包含的片段数量。当用户询问“我的知识库里有什么”“我记录过哪些内容”或者
        在不确定某个主题是否已有记录时，可调用此工具获得全貌。

        Args:
            无参数。

        Returns:
            以 JSON 字符串返回文档列表，每条包含：
              - doc_name: 文档名（文件名）
              - tags: 该文档的章节标签列表
              - updated_at: 最近更新时间
              - chunk_count: 片段数量
        """
        if not store:
            raise ValueError("list_documents 未绑定 vector store，无法执行。")
        try:
            items = store.all_documents()
        except Exception as exc:
            logger.exception("list_documents failed")
            return f"查询失败: {exc}"

        # Aggregate per source_doc.
        by_doc: dict[str, dict[str, Any]] = {}
        for item in items:
            meta = item.get("metadata") or {}
            name = meta.get("source_doc", "unknown")
            entry = by_doc.setdefault(
                name,
                {"doc_name": name, "tags": [], "updated_at": "", "chunk_count": 0},
            )
            section = meta.get("section_header", "")
            if section and section not in entry["tags"]:
                entry["tags"].append(section)
            entry["chunk_count"] += 1
            ts = meta.get("created_at", "")
            if ts and (not entry["updated_at"] or ts > entry["updated_at"]):
                entry["updated_at"] = ts

        docs = sorted(by_doc.values(), key=lambda d: d["doc_name"])
        return json.dumps(docs, ensure_ascii=False)

    return list_documents


__all__ = ["_make_list_documents"]
