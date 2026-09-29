"""`create_note` tool —— 持久化一条新 markdown 笔记并立即建索引.

笔记写入 `./data/notes/<sanitized-title>.md`,并在同一次调用中完成 embed 与
加入 Chroma collection,因此随后的 `knowledge_search` 调用能立即检索到它.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _make_create_note(
    store: Any | None,
    notes_dir: Path | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
):
    """闭包工厂,使 tool 在运行时绑定到 store 与 notes 目录."""

    @tool
    def create_note(title: str, content: str) -> str:
        """创建一条新的知识笔记，写入本地 markdown 文件并立刻加入知识库检索。

        当用户想记录一个新想法、会议结论、学习心得或任何值得保存的内容时调用。
        调用成功后，该笔记会立即可被知识搜索检索到，无需重建索引。

        Args:
            title: 笔记标题。建议简洁且能概括内容，例如“2026 Q2 技术复盘”。会用作文件名。
            content: 笔记正文，支持 markdown 格式。应包含用户要求的完整内容。

        Returns:
            JSON 字符串：{"status": "success"|"error", "path": 文件路径, "chunks": 索引片段数}
        """
        if store is None:
            raise ValueError("create_note 未绑定 vector store，无法执行。")
        title = (title or "").strip()
        content = (content or "").strip()
        if not title:
            return json.dumps({"status": "error", "message": "标题不能为空"}, ensure_ascii=False)
        if not content:
            return json.dumps({"status": "error", "message": "内容不能为空"}, ensure_ascii=False)

        base_dir = Path(notes_dir) if notes_dir else Path("./data/notes")
        base_dir.mkdir(parents=True, exist_ok=True)

        filename = _sanitize_filename(title) + ".md"
        path = base_dir / filename
        try:
            path.write_text(
                f"# {title}\n\n{content}\n",
                encoding="utf-8",
            )
        except OSError as exc:
            logger.exception("Failed to write note %s", path)
            return json.dumps({"status": "error", "message": f"写入文件失败: {exc}"},
                              ensure_ascii=False)

        # 立即建索引到 Chroma(识别 markdown 标题 → 作为一个 section).
        doc = Document(
            page_content=f"# {title}\n\n{content}",
            metadata={
                "source_doc": filename,
                "path": str(path),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        try:
            from rag.splitter import split_documents

            chunks = split_documents(
                [doc], chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            # 用已存在的标题重存笔记会覆盖文件,因此必须替换旧的 chunk 而不是
            # 追加(否则它们会以新的 id 残留为过期重复片段).
            # `reindex_path` 以文件 *path* 为键——若以文件名作键,会同时匹配
            # 知识库目录中的同名文档并把它清掉——而且它先新增再删除,所以这里
            # 即使失败也不会丢失已建索引的笔记.
            n = store.reindex_path(str(path), chunks)
        except Exception as exc:
            # 即使建索引失败,文件也已写入 —— 必须明确告知用户.
            logger.exception("Note written but indexing failed")
            return json.dumps(
                {"status": "error",
                "message": f"文件已写入但索引失败: {exc}",
                "path": str(path)},
                ensure_ascii=False,
            )

        return json.dumps(
            {"status": "success", "path": str(path), "chunks": n},
            ensure_ascii=False,
        )

    return create_note


def _sanitize_filename(title: str) -> str:
    """由标题生成文件系统安全的文件主干名(保留中日韩字符)."""
    stem = re.sub(r"[\\/:*?\"<>|]+", "_", title).strip().strip(".")
    # 合并连续空白/下划线;并限制长度.
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"_+", "_", stem)[:80]
    return stem or "note"


__all__ = ["_make_create_note"]
