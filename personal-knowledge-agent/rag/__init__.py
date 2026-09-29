"""RAG 流水线:load → split → embed → index → retrieve → rerank.

对包内其余部分公开的接口:

    - `load_documents`:  从目录读取 .md/.txt 文件
    - `split_documents`: 感知 markdown 标题的 chunk 切分,带兜底策略
    - `Embedder`:        对 `embedding-3` API 的批处理封装
    - `VectorStore`:     持久化的 ChromaDB store,含 metadata
    - `Retriever`:       带 relevance score 的 top-k 相似度检索
    - `Reranker`:        可选的 cross-encoder rerank(可优雅降级)

所有内容都可直接从 `rag` 导入,例如 `from rag import Retriever`.
"""

from .embedder import Embedder
from .loader import load_documents
from .reranker import Reranker
from .retriever import Retriever
from .splitter import split_documents
from .vectorstore import VectorStore

__all__ = [
    "Embedder",
    "load_documents",
    "split_documents",
    "Reranker",
    "Retriever",
    "VectorStore",
]
