"""Central configuration for the personal knowledge agent.

Loaded from environment variables (see .env.example). The Settings object is
created once at import time, so every module (`rag`, `tools`, `graph`) reads
from the same source of truth. `python-dotenv` is wired into pydantic-settings,
so a `.env` file placed next to this module is loaded automatically.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = directory containing this file.
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# If the user hasn't pinned a Hugging Face endpoint, default to the China
# mirror so local model downloads (embedding / reranker) work behind the
# Great Firewall. Set HF_ENDPOINT yourself to override.
if not os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class Settings(BaseSettings):
    """Runtime configuration. All values can be overridden via environment."""

    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), extra="ignore")

    # --- GLM-5.2 ---------------------------------------------------------
    zhipu_api_key: str = os.getenv("ZHIPU_API_KEY", "")
    llm_model: str = "glm-5.2"
    embedding_model: str = "embedding-3"
    # Embedding backend: "zhipu" (OpenAI-compatible API, needs embedding
    # resource package) or "local" (sentence-transformers, fully offline).
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "zhipu")
    # Local embedding model used when EMBEDDING_BACKEND=local.
    local_embedding_model: str = os.getenv(
        "LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"
    )
    # OpenAI-compatible endpoint exposed by Zhipu BigModel.
    openai_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    # Fallback in case the model id changes upstream.
    chat_model_alias: str = "glm-5.2"

    # --- Storage paths ----------------------------------------------------
    kb_dir: Path = BASE_DIR / "data" / "kb"
    notes_dir: Path = BASE_DIR / "data" / "notes"
    chroma_db_dir: Path = BASE_DIR / "data" / "chroma"
    chroma_collection_name: str = "personal_knowledge"
    sqlite_checkpoint_path: Path = BASE_DIR / "data" / "checkpoints.sqlite"

    # --- RAG tuning -------------------------------------------------------
    top_k: int = 10
    rerank_top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 64
    embedding_batch_size: int = 20

    # --- Reranker ---------------------------------------------------------
    reranker_enabled: bool = True
    reranker_model: str = "bge-reranker-v2-m3"
    reranker_device: str = "cpu"  # "cpu" | "mps" | "cuda"

    # --- Graph behaviour --------------------------------------------------
    max_history_messages: int = 8
    max_retry: int = 1
    # Upper bound on model→tool→model round trips inside one generation step.
    max_tool_iterations: int = 3
    # Reserved, currently inert: `ZhipuLLM` supports a `stream_tokens`
    # callback, but nothing passes one in — `main.py` does not forward this to
    # `build_graph`, so the CLI still prints whole answers at the end.
    stream_tokens: bool = True

    # --- API error codes (GLM-specific) -----------------------------------
    err_code_insufficient_balance: int = 1301
    err_code_rate_limit: int = 1305

    @property
    def resolve_model_name(self) -> str:
        """The model id to send to the API. Kept as a property so a single
        override (LLM_MODEL env var) drives every call."""
        return self.llm_model or self.chat_model_alias

    @property
    def _resolved(self) -> dict:
        # helper used by __post_init__-style validation below
        return {
            "kb_dir": self.kb_dir,
            "notes_dir": self.notes_dir,
            "chroma_db_dir": self.chroma_db_dir,
        }

    def ensure_dirs(self) -> None:
        """Create every on-disk directory this agent needs."""
        for path in (self.kb_dir, self.notes_dir, self.chroma_db_dir,
                     self.sqlite_checkpoint_path.parent):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the (cached) global settings instance."""
    return Settings()


def validate_api_key(settings: Settings) -> None:
    """Fail fast with a clear message if the API key is missing."""
    if not settings.zhipu_api_key:
        raise RuntimeError(
            "ZHIPU_API_KEY is not set. Copy .env.example to .env and fill in "
            "your key from https://open.bigmodel.cn."
        )


def _device_or_default(device: str | None) -> Literal["cpu", "mps", "cuda"]:
    """Normalize the reranker device string."""
    if device and device.lower() in {"cpu", "mps", "cuda"}:
        return device.lower()  # type: ignore[return-value]
    return "cpu"


__all__ = ["Settings", "get_settings", "validate_api_key", "BASE_DIR"]
