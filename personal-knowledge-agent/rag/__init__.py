"""RAG pipeline: load → split → embed → index → retrieve → rerank.

Public surface exposed to the rest of the package:

    - `load_documents`:  read .md/.txt files from a directory
    - `split_documents`: markdown-header-aware chunking with fallback
    - `Embedder`:        batching wrapper around the `embedding-3` API
    - `VectorStore`:     persistent ChromaDB store with metadata
    - `Retriever`:       top-k similarity search with relevance scores
    - `Reranker`:        optional cross-encoder rerank (graceful fallback)

Everything is importable from `rag` directly, e.g. `from rag import Retriever`.
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
