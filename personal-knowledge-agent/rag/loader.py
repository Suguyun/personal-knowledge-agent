"""从磁盘加载知识库文档到 langchain Documents.

支持的扩展名:`.md` 和 `.txt`.目录中其余文件会被跳过并记录一条日志.
文件以 UTF-8 读取,失败时回退到 latin-1,因此单个编码错误的文件不会中断
整个 index 的构建.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt"}
# 当前目录标记(macOS 的 `._` 文件,Finder 产物)——
# 永远不会被当作知识内容.
_SKIP_PREFIXES = (".", "_")


def load_documents(kb_dir: str | Path) -> list[Document]:
    """将 `kb_dir` 下(非递归)所有受支持的文件读入 Documents.

    Args:
        kb_dir: 存放知识库 markdown/text 文件的目录.

    Returns:
        langchain `Document` 对象的列表.`metadata` 包含:
        `source_doc`(文件名),`path`(绝对文件路径)以及
        `created_at`(文件 mtime 的 RFC3339 UTC 时间戳).
    """
    root = Path(kb_dir)
    if not root.is_dir():
        logger.warning("Knowledge base directory %s does not exist.", root)
        return []

    documents: list[Document] = []
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            logger.debug("Skipping unsupported file: %s", path.name)
            continue
        if path.name.startswith(_SKIP_PREFIXES):
            logger.debug("Skipping hidden file: %s", path.name)
            continue

        content = _read_text(path)
        if not content or not content.strip():
            logger.debug("Skipping empty file: %s", path.name)
            continue

        mtime = path.stat().st_mtime
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source_doc": path.name,
                    "path": str(path),
                    "created_at": _iso_utc(mtime),
                },
            )
        )
        logger.info("Loaded %s (%d chars)", path.name, len(content))

    logger.info("Loaded %d document(s) from %s", len(documents), root)
    return documents


def _read_text(path: Path) -> str:
    """以 UTF-8 读取,对旧文件回退到 latin-1."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("UTF-8 decode failed for %s, using latin-1.", path.name)
        return path.read_text(encoding="latin-1")


def _iso_utc(epoch: float) -> str:
    """把 unix 时间戳格式化为 RFC3339 UTC 字符串."""
    import datetime as _dt

    return _dt.datetime.fromtimestamp(epoch, tz=_dt.timezone.utc).isoformat()


__all__ = ["load_documents"]
