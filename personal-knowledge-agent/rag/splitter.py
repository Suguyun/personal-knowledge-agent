"""感知 Markdown 的文档切分.

策略(按规格):先尝试 `MarkdownHeaderTextSplitter` —— 它保留小节标题,
使每个 chunk 都带有一个可供 LLM 引用的 `section_header` metadata 字段.
标题切分后剩余的内容(首个标题前的普通文本,列表项等)再用
`RecursiveCharacterTextSplitter` 兜底切分,确保不丢内容.

最后所有 chunk 都会经过一次字符级切分做归一化,保证没有 chunk 超过
`chunk_size` 个字符.
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
    """把一组 Documents 切分成带丰富 metadata 的检索 chunk.

    每个输出 chunk 携带的 metadata:

        - `source_doc`    : 原始文件名(继承而来)
        - `section_header`: chunk 所属的 markdown 标题路径
        - `chunk_index`   : 在返回的 chunk 列表中的 0 起始下标.它跨文档连续
                            计数,因此并不是 `source_doc` 内的下标.
        - `created_at`    : 继承自源文档

    Args:
        documents: 来自 `load_documents` 的原始文档.
        chunk_size: 每个 chunk 的目标最大字符数.
        chunk_overlap: 相邻兜底 chunk 之间的 overlap.

    Returns:
        扁平的 `Document` chunk 列表.
    """
    # MarkdownHeaderTextSplitter 会把匹配到的标题文本存放在下面这些
    # label 键下,并去掉 "#" 标记
    # (例如 piece.metadata["H1"] == "工具选型").
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
    # 最后的安全网:即使标题切分产生了过大的小节正文,
    # 也绝不超出 chunk_size 个字符.
    hard_cap = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )

    chunks: list[Document] = []
    for doc in documents:
        source_meta = {k: v for k, v in doc.metadata.items() if k not in header_labels}
        header_chunks = header_splitter.split_text(doc.page_content)

        # 标题 splitter 只会返回落在某个标题下的文档.
        # 如果文件完全没有标题,我们会得到空列表 —— 这正是
        # 兜底 splitter 覆盖的场景.
        if not header_chunks:
            logger.debug("No headers in %s, using recursive splitter.",
                         source_meta.get("source_doc", "?"))
            header_chunks = [
                Document(page_content=doc.page_content, metadata={})
            ]

        for piece in header_chunks:
            # 根据标题 splitter 填充的 label 键重建 "H1 / H2 / H3",
            # 例如 "# AI 编码工具 / ## 使用技巧".
            section = _extract_section(piece.metadata, header_labels)
            # 对仍然超出大小上限的正文再次切分,并在每个片段上保留
            # 小节标题,以便仍可引用.
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
    """根据标题 splitter 的 label 重建 "H1 / H2" 引用路径.

    `MarkdownHeaderTextSplitter` 会去掉 "#" 标记,因此 metadata["H1"] 形如
    "AI 编码工具",metadata["H2"] 形如 "使用技巧"(已对照示例 KB 验证:
    位于 `# 2026 年度 OKR` / `## 年度目标` 下的 chunk 得到的
    section_header 为 "2026 年度 OKR / 年度目标").我们用 " / " 拼接存在
    的各层级,使 section_header 读起来自然.
    """
    parts = [metadata.get(label, "").strip() for label in labels]
    parts = [p for p in parts if p]
    return " / ".join(parts)


def _build_chunk(text: str, source_meta: dict[str, Any],
                 section: str, index: int) -> Document:
    """按完整的 metadata 契约组装单个 chunk Document."""
    metadata: dict[str, Any] = {
        "source_doc": source_meta.get("source_doc", "unknown"),
        "section_header": section,
        "chunk_index": index,
        "created_at": source_meta.get("created_at", ""),
    }
    # 保留原始绝对路径(若存在),供 create_note 审计使用.
    if source_meta.get("path"):
        metadata["path"] = source_meta["path"]
    return Document(page_content=text, metadata=metadata)


__all__ = ["split_documents"]
