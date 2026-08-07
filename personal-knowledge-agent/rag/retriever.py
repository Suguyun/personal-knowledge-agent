"""Retrieval orchestration: embed query → top-k search → optional rerank.

`Retriever` is the single entry point the `knowledge_search` tool uses. It:

    1. embeds the (possibly rewritten) query,
    2. fetches `top_k` candidates with relevance scores,
    3. reranks them with the cross-encoder (when enabled),
    4. returns `rerank_top_k` results, never silently dropping to zero.

If the reranker is disabled or fails to load, it degrades gracefully to the
raw top-k results and logs a warning — retrieval still works.
"""

from __future__ import annotations

import logging
from typing import Any

from config import Settings, get_settings

logger = logging.getLogger(__name__)


class Retriever:
    """Query → scored, reranked knowledge chunks."""

    def __init__(
        self,
        settings: Settings | None = None,
        store: Any | None = None,
        embedder: Any | None = None,
        reranker: Any | None = None,
    ) -> None:
        """Wire the retriever to a store, embedder and (optional) reranker.

        Args:
            settings: App settings.
            store: A `VectorStore` instance (created here if omitted).
            embedder: An object exposing `embed_texts(list[str])`.
            reranker: A `Reranker` instance (created here if omitted).
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
        """Embed `query`, retrieve `top_k` chunks, then rerank.

        Args:
            query: Natural-language search query (may be a rewritten query).
            top_k: Number of candidates to fetch (defaults to settings.top_k).
            filters: Optional metadata filter passed straight to Chroma.

        Returns:
            List of dicts `{"content", "metadata", "score"}` sorted by
            relevance, length `min(rerank_top_k, candidates)`.
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
        """Fallback path: raw similarity scores, truncated to n."""
        return sorted(hits, key=lambda h: h["score"], reverse=True)[:n]

    def _warn_once(self, message: str) -> None:
        """Emit a degradation warning only once per process."""
        if not self._fallback_warned:
            logger.warning(message)
            self._fallback_warned = True


__all__ = ["Retriever"]
