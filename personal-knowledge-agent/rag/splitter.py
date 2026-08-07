"""Markdown-aware document splitting.

Strategy (per spec): try `MarkdownHeaderTextSplitter` first — it preserves the
section header so each chunk keeps a `section_header` metadata field that the
LLM can cite. Any content left over after header splitting (plain prose before
the first header, list items, etc.) is then chunked with a
`RecursiveCharacterTextSplitter` fallback so nothing is dropped.

All chunks are normalized afterwards with a final character-level split so no
chunk ever exceeds `chunk_size` characters.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

logger = logging.getLogger(__name__)


def split_documents(
    documents: Iterable[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Document]:
    """Split a list of Documents into retrieval chunks with rich metadata.

    Each output chunk carries metadata:

        - `source_doc`    : original file name (inherited)
        - `section_header`: markdown heading the chunk lives under
        - `chunk_index`   : 0-based order within the source document
        - `created_at`    : inherited from the source document

    Args:
        documents: Raw documents from `load_documents`.
        chunk_size: Target max characters per chunk.
        chunk_overlap: Overlap between consecutive fallback chunks.

    Returns:
        A flat list of `Document` chunks.
    """
    # MarkdownHeaderTextSplitter stores the matched header text under the
    # label keys below (e.g. piece.metadata["H1"] == "# 工具选型").
    header_labels = ("H1", "H2", "H3")
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")],
        strip_headers=False,
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )
    # Final safety net: never exceed chunk_size characters, even after header
    # splitting produced an oversized section body.
    hard_cap = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )

    chunks: list[Document] = []
    for doc in documents:
        source_meta = {k: v for k, v in doc.metadata.items() if k not in header_labels}
        header_chunks = header_splitter.split_text(doc.page_content)

        # The header splitter only returns documents that fall under a header.
        # If a file has no headers at all we get an empty list — that is the
        # exact case the fallback splitter covers.
        if not header_chunks:
            logger.debug("No headers in %s, using recursive splitter.",
                         source_meta.get("source_doc", "?"))
            header_chunks = [
                Document(page_content=doc.page_content, metadata={})
            ]

        for piece in header_chunks:
            # Reconstruct "H1 / H2 / H3" from the label keys the header
            # splitter populated, e.g. "# AI 编码工具 / ## 使用技巧".
            section = _extract_section(piece.metadata, header_labels)
            # Re-chunk any body that still exceeds the size cap, preserving the
            # section header on each fragment so citation stays possible.
            if len(piece.page_content) > chunk_size:
                fragments = char_splitter.split_text(piece.page_content)
            else:
                fragments = [piece.page_content]

            for i, text in enumerate(fragments):
                if len(text) > chunk_size:
                    for j, capped in enumerate(hard_cap.split_text(text)):
                        chunks.append(
                            _build_chunk(capped, source_meta, section, len(chunks))
                        )
                else:
                    chunks.append(
                        _build_chunk(text, source_meta, section, len(chunks))
                    )

    logger.info("Split into %d chunks (chunk_size=%d, overlap=%d)",
                len(chunks), chunk_size, chunk_overlap)
    return chunks


def _extract_section(metadata: dict[str, Any], labels: tuple[str, ...]) -> str:
    """Reconstruct a "H1 / H2" citation path from the header splitter labels.

    `MarkdownHeaderTextSplitter` sets e.g. metadata["H1"] = "# AI 编码工具",
    metadata["H2"] = "## 使用技巧". We join the present levels with " / " so
    the section_header reads naturally: "# AI 编码工具 / ## 使用技巧".
    """
    parts = [metadata.get(label, "").strip() for label in labels]
    parts = [p for p in parts if p]
    return " / ".join(parts)


def _build_chunk(text: str, source_meta: dict[str, Any],
                 section: str, index: int) -> Document:
    """Assemble a single chunk Document with the full metadata contract."""
    metadata: dict[str, Any] = {
        "source_doc": source_meta.get("source_doc", "unknown"),
        "section_header": section,
        "chunk_index": index,
        "created_at": source_meta.get("created_at", ""),
    }
    # Preserve the original absolute path (if present) for create_note audits.
    if source_meta.get("path"):
        metadata["path"] = source_meta["path"]
    return Document(page_content=text, metadata=metadata)


__all__ = ["split_documents"]
