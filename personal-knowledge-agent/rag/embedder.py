"""Embeddings backend.

Two backends are supported, selected by `EMBEDDING_BACKEND`:

- `zhipu` (default): Zhipu `embedding-3` over the OpenAI-compatible API,
  batched in groups of `embedding_batch_size`. Requires an embedding resource
  package on the account.
- `local`: a `sentence-transformers` model (e.g. `BAAI/bge-small-zh-v1.5`)
  loaded on-device. Fully offline, no API quota, no cost. Great for getting
  started when the Zhipu account has chat quota but no embedding package.

`Embedder` owns the lazy backend client and exposes a thin `embed_texts`
interface so callers (index builder + query path) never touch backend details.
"""

from __future__ import annotations

import logging

from config import Settings, get_settings

logger = logging.getLogger(__name__)


class Embedder:
    """Embedding client with swappable backend (zhipu API or local model)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._zhipu_client: object | None = None
        self._local_model: object | None = None

    @property
    def backend(self) -> str:
        """Normalized backend name (zhipu | local)."""
        return (self.settings.embedding_backend or "zhipu").strip().lower()

    @property
    def client(self):
        """Lazily-initialized Zhipu OpenAI-compatible client (zhipu backend)."""
        if self._zhipu_client is None:
            from openai import OpenAI

            self._zhipu_client = OpenAI(
                api_key=self.settings.zhipu_api_key,
                base_url=self.settings.openai_base_url,
            )
        return self._zhipu_client

    @property
    def local_model(self):
        """Lazily-loaded sentence-transformers model (local backend)."""
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading local embedding model %s …",
                        self.settings.local_embedding_model)
            self._local_model = SentenceTransformer(self.settings.local_embedding_model)
        return self._local_model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts; returns one float vector per input, in order."""
        if not texts:
            return []

        if self.backend == "local":
            return self._embed_local(texts)

        return self._embed_zhipu(texts)

    # --- zhipu backend ------------------------------------------------------
    def _embed_zhipu(self, texts: list[str]) -> list[list[float]]:
        batch_size = self.settings.embedding_batch_size
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(
                model=self.settings.embedding_model,
                input=batch,
            )
            # The API returns items in request order; sort defensively by index
            # so ordering never silently depends on upstream behaviour.
            ordered = sorted(response.data, key=lambda item: item.index)
            results.extend([item.embedding for item in ordered])
        logger.debug("Embedded %d text(s) via zhipu in %d batch(es).",
                     len(texts), (len(texts) + batch_size - 1) // batch_size)
        return results

    # --- local backend ------------------------------------------------------
    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        batch_size = self.settings.embedding_batch_size
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors = self.local_model.encode(batch, normalize_embeddings=True)
            # encode() returns a numpy array; normalize to a python float list
            # so the return contract stays identical across backends.
            results.extend([v.tolist() for v in vectors])
        logger.debug("Embedded %d text(s) via local model %s.",
                     len(texts), self.settings.local_embedding_model)
        return results


__all__ = ["Embedder"]
