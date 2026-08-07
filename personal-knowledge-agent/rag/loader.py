"""Load knowledge base documents from disk into langchain Documents.

Supported extensions: `.md` and `.txt`. Every other file in the directory is
skipped with a log message. Files are read as UTF-8 with a latin-1 fallback so
a single mis-encoded file cannot abort the whole index build.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt"}
# Current-working-directory marker (macOS `._` files, Finder artifacts) —
# never treated as knowledge content.
_SKIP_PREFIXES = (".", "_")


def load_documents(kb_dir: str | Path) -> list[Document]:
    """Read every supported file under `kb_dir` (non-recursive) into Documents.

    Args:
        kb_dir: Directory containing the knowledge base markdown/text files.

    Returns:
        A list of langchain `Document` objects. `metadata` carries:
        `source_doc` (file name), `path` (absolute file path) and
        `created_at` (RFC3339 UTC timestamp of the file's mtime).
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
    """Read UTF-8, falling back to latin-1 for legacy files."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("UTF-8 decode failed for %s, using latin-1.", path.name)
        return path.read_text(encoding="latin-1")


def _iso_utc(epoch: float) -> str:
    """Format a unix timestamp as an RFC3339 UTC string."""
    import datetime as _dt

    return _dt.datetime.fromtimestamp(epoch, tz=_dt.timezone.utc).isoformat()


__all__ = ["load_documents"]
