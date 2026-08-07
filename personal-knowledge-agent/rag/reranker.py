"""Optional cross-encoder reranker over the top-k retrieval results.

Uses `bge-reranker-v2-m3` via `sentence-transformers` CrossEncoder. The model
is downloaded on first use (~2GB). Loading happens lazily on the first `rerank`
call so an app that never retrieves (or runs with reranking disabled) pays no
startup cost.

Degradation contract (per spec): if the reranker is unavailable or fails, the
caller (`Retriever`) falls back to raw retrieval scores. This module only ever
raises during construction; `rerank` catches runtime failures internally.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from config import Settings, get_settings

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker with lazy model loading."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model: Any | None = None
        self._load_error: str | None = None

    def _ensure_model(self) -> Any:
        """Load the cross-encoder once; raise a clear error if impossible."""
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            raise RuntimeError(self._load_error)

        if not self.settings.reranker_enabled:
            raise RuntimeError("Reranker disabled via RERANKER_ENABLED=false.")

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            self._load_error = (
                "sentence-transformers not installed; run "
                "`pip install sentence-transformers` or set "
                "RERANKER_ENABLED=false."
            )
            raise RuntimeError(self._load_error) from exc

        # bge-reranker-v2-m3 was trained with the query/passage instruction
        # prefix; applying it improves ranking accuracy on zh/en mixed content.
        # `max_length` truncates long chunks so the model accepts everything.
        self._model = CrossEncoder(
            model_name=self.settings.reranker_model,
            device=self.settings.reranker_device,
            max_length=512,
        )
        logger.info("Loaded reranker %s on %s.",
                    self.settings.reranker_model, self.settings.reranker_device)
        return self._model

    def rerank(
        self,
        query: str,
        hits: list[dict[str, Any]],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """Rerank `hits` (already containing `content`) against `query`.

        Args:
            query: The user query (or rewritten query).
            hits: Candidate chunks from the vector store, each with `content`.
            top_n: Number of results to keep after reranking.

        Returns:
            Reordered `hits` (original dicts, score updated to the sigmoid
            cross-encoder score) truncated to `top_n`. Empty list on any
            failure — callers degrade to raw retrieval in that case.
        """
        if not hits:
            return []

        try:
            model = self._ensure_model()
        except Exception as exc:
            logger.warning("Reranker unavailable (%s); using raw retrieval.", exc)
            return []

        try:
            pairs = [
                [_PREFIX + query, hit["content"][:512]] for hit in hits
            ]
            scores = model.predict(pairs, show_progress_bar=False)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Rerank failed (%s); using raw retrieval.", exc)
            return []

        scored = list(zip(hits, scores))
        scored.sort(key=lambda item: float(item[1]), reverse=True)
        # Cross-encoder scores are raw logits (roughly ±5); squash to [0,1] so
        # downstream score handling stays uniform with vector similarity.
        top = scored[:top_n]
        return [
            {**hit, "score": round(1.0 / (1.0 + math.exp(-float(s))), 4)}
            for hit, s in top
        ]


_PREFIX = "为这个句子生成表示以用于检索相关文章："


__all__ = ["Reranker"]
