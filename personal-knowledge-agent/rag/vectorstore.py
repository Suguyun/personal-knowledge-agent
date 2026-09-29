"""Persistent ChromaDB vector store.

The store owns the collection lifecycle: it creates the collection once,
deletes-and-rebuilds on explicit `rebuild`, and exposes `add_documents` /
`query` / `all_documents` used by the retriever and the index build.

Chroma stores the per-chunk metadata dict directly, so filters passed to
`query` can reference any metadata field (e.g. `{"source_doc": "notes.md"}`).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Iterable

import chromadb
from langchain_core.documents import Document

from config import Settings, get_settings

logger = logging.getLogger(__name__)

# Chroma stores text as an official field; everything else we keep in metadata.
_TEXT_KEY = "text"
_SOURCE_KEY = "source_doc"
# File location — unique across data/kb/ and data/notes/, unlike source_doc.
_PATH_KEY = "path"


class VectorStore:
    """Thin wrapper over a persistent ChromaDB collection."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Any | None = None,
    ) -> None:
        """Create/attach to the persistent collection.

        Args:
            settings: App settings (defaults to the global instance).
            embedder: An object exposing `embed_texts(list[str]) -> vectors`.
                      Passed in so the retriever and indexer share one client.
        """
        self.settings = settings or get_settings()
        # Imported here to avoid a circular import at module load time.
        from .embedder import Embedder

        self.embedder = embedder or Embedder(self.settings)
        self._client = chromadb.PersistentClient(
            path=str(self.settings.chroma_db_dir)
        )
        self._collection = self._client.get_or_create_collection(
            name=self.settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self):
        """Access to the raw Chroma collection (advanced use)."""
        return self._collection

    def add_documents(self, chunks: Iterable[Document]) -> int:
        """Embed and insert document chunks.

        Args:
            chunks: Documents whose `page_content` is embedded and whose
                    `metadata` is stored alongside.

        Returns:
            Number of chunks inserted.
        """
        chunk_list = list(chunks)
        if not chunk_list:
            return 0

        texts = [c.page_content for c in chunk_list]
        embeddings = self.embedder.embed_texts(texts)

        ids = [str(uuid.uuid4()) for _ in chunk_list]
        metadatas = []
        for c in chunk_list:
            meta = dict(c.metadata)
            meta[_TEXT_KEY] = c.page_content  # always retain retrievable text
            # No size field: a character count used to be stored under
            # "n_tokens" (never read by anything, and mislabeled). Dropped
            # rather than renamed so the collection cannot end up carrying two
            # different key names for the same thing.
            metadatas.append(meta)

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info("Indexed %d chunk(s) into %s.",
                    len(chunk_list), self.settings.chroma_collection_name)
        return len(chunk_list)

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search the collection, returning scored hits.

        Args:
            query_embedding: Vector of the query text.
            n_results: Number of neighbours to fetch.
            where: Optional Chroma metadata filter (e.g. by source_doc).

        Returns:
            List of dicts: `{"content", "metadata", "score"}`.
            `score` is a similarity in [0, 1] — higher is more relevant.
        """
        kwargs: dict[str, Any] = dict(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        result = self._collection.query(**kwargs)

        # Chroma returns nested lists for each query; we sent exactly one.
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]

        hits: list[dict[str, Any]] = []
        for content, meta, dist in zip(docs, metas, dists):
            # Cosine distance ∈ [0, 2] → similarity ∈ [0, 1].
            score = max(0.0, min(1.0, 1.0 - float(dist)))
            meta = dict(meta) if meta else {}
            meta.pop(_TEXT_KEY, None)  # keep payloads lean for the LLM
            hits.append({"content": content, "metadata": meta, "score": score})
        return hits

    def all_documents(self) -> list[dict[str, Any]]:
        """Return every chunk with metadata (used by list_documents / stats).

        Returns:
            List of dicts: `{"content", "metadata"}`.
        """
        result = self._collection.get(include=["documents", "metadatas"])
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        out: list[dict[str, Any]] = []
        for content, meta in zip(docs, metas):
            meta = dict(meta) if meta else {}
            meta.pop(_TEXT_KEY, None)
            out.append({"content": content, "metadata": meta})
        return out

    def count(self) -> int:
        """Number of chunks currently in the collection."""
        return self._collection.count()

    def delete_all(self) -> None:
        """Wipe every chunk (used before a full rebuild)."""
        self._collection.delete(where={})
        logger.info("Cleared collection %s.", self.settings.chroma_collection_name)

    def reindex_path(self, path: str, chunks: Iterable[Document]) -> int:
        """Embed + insert `chunks`, replacing whatever was indexed for `path`.

        Keyed on the `path` metadata (where the file actually lives), **not**
        `source_doc` (the bare file name): the same name can exist in both
        `data/kb/` and `data/notes/`, so deleting by name would silently wipe
        the other document's vectors.

        Ordering matters. The new chunks are added first and the previous ones
        removed only once that succeeded — a failure (embedding backend down,
        network drop) then leaves the old content retrievable instead of
        deleting it and writing nothing.

        Args:
            path: Absolute-or-relative path recorded in the chunk metadata.
            chunks: Replacement chunks for that path.

        Returns:
            Number of chunks inserted.
        """
        previous_ids = self._ids_for_path(path)
        added = self.add_documents(chunks)
        if previous_ids:
            self.delete_ids(previous_ids)
        return added

    def _ids_for_path(self, path: str) -> list[str]:
        """Ids of every chunk currently stored for `path`."""
        result = self._collection.get(where={_PATH_KEY: path})
        return list(result.get("ids") or [])

    def delete_ids(self, ids: Iterable[str]) -> None:
        """Delete chunks by id (no-op for an empty list)."""
        id_list = list(ids)
        if id_list:
            self._collection.delete(ids=id_list)

    def reset(self) -> None:
        """Drop and recreate the collection from scratch."""
        try:
            self._client.delete_collection(self.settings.chroma_collection_name)
        except Exception:
            logger.debug("Collection did not exist; creating fresh.")
        self._collection = self._client.get_or_create_collection(
            name=self.settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Reset collection %s.", self.settings.chroma_collection_name)


__all__ = ["VectorStore", "_SOURCE_KEY", "_PATH_KEY"]
