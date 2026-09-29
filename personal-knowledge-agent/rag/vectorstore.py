"""持久化的 ChromaDB vector store.

store 负责 collection 的生命周期:只创建一次 collection,在显式 `rebuild`
时删除并重建,并对外提供 retriever 与 index 构建所用的 `add_documents` /
`query` / `all_documents`.

Chroma 直接存储每个 chunk 的 metadata dict,因此传给 `query` 的过滤条件
可以引用任意 metadata 字段(例如 `{"source_doc": "notes.md"}`).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Iterable

import chromadb
from langchain_core.documents import Document

from config import Settings, get_settings

logger = logging.getLogger(__name__)

# Chroma 把文本作为官方字段存储;其余内容我们放进 metadata.
_TEXT_KEY = "text"
_SOURCE_KEY = "source_doc"
# 文件位置 —— 与 source_doc 不同,它在 data/kb/ 和 data/notes/ 之间唯一.
_PATH_KEY = "path"


class VectorStore:
    """对持久化 ChromaDB collection 的轻量封装."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Any | None = None,
    ) -> None:
        """创建或附加到持久化 collection.

        Args:
            settings: 应用配置(默认使用全局实例).
            embedder: 暴露 `embed_texts(list[str]) -> vectors` 的对象.
                      传入它是为了让 retriever 与 indexer 共用一个 client.
        """
        self.settings = settings or get_settings()
        # 在此处导入,避免模块加载时的循环导入.
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
        """访问原生 Chroma collection(进阶用法)."""
        return self._collection

    def add_documents(self, chunks: Iterable[Document]) -> int:
        """对文档 chunk 做 embedding 并插入.

        Args:
            chunks: 其 `page_content` 会被 embedding,`metadata` 会被一并
                    存储的 Documents.

        Returns:
            插入的 chunk 数量.
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
            meta[_TEXT_KEY] = c.page_content  # 始终保留可检索文本
            # 没有 size 字段:字符数过去存放在 "n_tokens" 下(从未被读取,
            # 且命名有误).这里直接删除而不是重命名,避免 collection 里为
            # 同一个东西出现两个不同的键名.
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
        """检索 collection,返回带分数的 hits.

        Args:
            query_embedding: 查询文本的向量.
            n_results: 取回的近邻数量.
            where: 可选的 Chroma metadata 过滤条件(例如按 source_doc).

        Returns:
            dict 列表:`{"content", "metadata", "score"}`.
            `score` 是 [0, 1] 区间的相似度 —— 越高越相关.
        """
        kwargs: dict[str, Any] = dict(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        result = self._collection.query(**kwargs)

        # Chroma 会为每个 query 返回嵌套列表;我们只发了恰好一个.
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]

        hits: list[dict[str, Any]] = []
        for content, meta, dist in zip(docs, metas, dists):
            # Cosine 距离 ∈ [0, 2] → 相似度 ∈ [0, 1].
            score = max(0.0, min(1.0, 1.0 - float(dist)))
            meta = dict(meta) if meta else {}
            meta.pop(_TEXT_KEY, None)  # 精简发给 LLM 的 payload
            hits.append({"content": content, "metadata": meta, "score": score})
        return hits

    def all_documents(self) -> list[dict[str, Any]]:
        """返回所有 chunk 及其 metadata(供 list_documents / stats 使用).

        Returns:
            dict 列表:`{"content", "metadata"}`.
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
        """collection 中当前的 chunk 数量."""
        return self._collection.count()

    def delete_all(self) -> None:
        """清空所有 chunk(全量重建前使用)."""
        self._collection.delete(where={})
        logger.info("Cleared collection %s.", self.settings.chroma_collection_name)

    def reindex_path(self, path: str, chunks: Iterable[Document]) -> int:
        """对 `chunks` 做 embedding 并插入,替换 `path` 已索引的内容.

        以 metadata 中的 `path`(文件实际所在位置)为键,而**不是**
        `source_doc`(裸文件名):同名文件可能同时存在于 `data/kb/` 和
        `data/notes/`,按名称删除会静默抹掉另一份文档的向量.

        顺序很重要.先添加新 chunk,成功后才删除旧的 —— 若过程中失败
        (embedding 后端宕机,网络中断),旧内容仍可检索,而不是先删后写
        导致什么都没留下.

        Args:
            path: chunk metadata 中记录的绝对或相对路径.
            chunks: 该 path 的替换 chunk.

        Returns:
            插入的 chunk 数量.
        """
        previous_ids = self._ids_for_path(path)
        added = self.add_documents(chunks)
        if previous_ids:
            self.delete_ids(previous_ids)
        return added

    def _ids_for_path(self, path: str) -> list[str]:
        """当前为 `path` 存储的所有 chunk 的 id."""
        result = self._collection.get(where={_PATH_KEY: path})
        return list(result.get("ids") or [])

    def delete_ids(self, ids: Iterable[str]) -> None:
        """按 id 删除 chunk(空列表时为 no-op)."""
        id_list = list(ids)
        if id_list:
            self._collection.delete(ids=id_list)

    def reset(self) -> None:
        """从头删除并重建 collection."""
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
