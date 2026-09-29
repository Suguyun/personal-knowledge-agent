"""检索编排:embed query → top-k search → 可选 rerank.

`Retriever` 是 `knowledge_search` 工具使用的唯一入口.它会:

    1. 对(可能已改写的)query 做 embedding,
    2. 取回带 relevance score 的 `top_k` 个候选,
    3. 用 cross-encoder 对它们做 rerank(若启用),
    4. 返回 `rerank_top_k` 条结果,绝不静默变成零条.

若 reranker 被禁用或加载失败,则优雅降级为原始 top-k 结果并记录一条
warning —— 检索仍然可用.
"""

from __future__ import annotations

import logging
from typing import Any

from config import Settings, get_settings

logger = logging.getLogger(__name__)


class Retriever:
    """查询 → 带分数,已 rerank 的知识 chunk."""

    def __init__(
        self,
        settings: Settings | None = None,
        store: Any | None = None,
        embedder: Any | None = None,
        reranker: Any | None = None,
    ) -> None:
        """把 retriever 接到 store,embedder 和(可选的)reranker.

        Args:
            settings: 应用配置.
            store: 一个 `VectorStore` 实例(省略时在此创建).
            embedder: 暴露 `embed_texts(list[str])` 的对象.
            reranker: 一个 `Reranker` 实例(省略时在此创建).
        """
        self.settings = settings or get_settings()
        from .vectorstore import VectorStore

        self.store = store or VectorStore(self.settings, embedder=embedder)

        from .reranker import Reranker

        self.reranker = reranker if reranker is not None else Reranker(self.settings)
        self._fallback_warned = False

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """对 `query` 做 embedding,检索 `top_k` 个 chunk,然后 rerank.

        Args:
            query: 自然语言检索 query(可能是改写后的 query).
            top_k: 取回的候选数量(默认取 settings.top_k).
            filters: 可选的 metadata 过滤条件,直接传给 Chroma.

        Returns:
            `{"content", "metadata", "score"}` 形式的 dict 列表,按相关度
            排序,长度为 `min(rerank_top_k, candidates)`.
        """
        top_k = top_k or self.settings.top_k
        query_embedding = self.store.embedder.embed_texts([query])[0]
        hits = self.store.query(query_embedding, n_results=top_k, where=filters)

        if not hits:
            logger.info("Retrieval returned no results for query: %r", query)
            return []

        if self.settings.reranker_enabled:
            reranked = self.reranker.rerank(query, hits, top_n=self.settings.rerank_top_k)
            if reranked:
                return reranked
            self._warn_once("Reranker produced no output; using raw retrieval.")

        return self._raw_top(hits, self.settings.rerank_top_k)

    def _raw_top(self, hits: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
        """兜底路径:原始 similarity score,截断到 n 条."""
        return sorted(hits, key=lambda h: h["score"], reverse=True)[:n]

    def _warn_once(self, message: str) -> None:
        """每个进程只发出一次降级 warning."""
        if not self._fallback_warned:
            logger.warning(message)
            self._fallback_warned = True


__all__ = ["Retriever"]
