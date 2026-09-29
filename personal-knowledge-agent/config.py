"""个人知识助手的集中配置.

从环境变量加载(见 .env.example).Settings 对象在 import 时只创建一次,因此每个
模块(`rag`,`tools`,`graph`)都读取同一个事实来源.`python-dotenv` 已接入
pydantic-settings,所以放在本模块旁的 `.env` 文件会自动加载.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录 = 包含本文件的目录.
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# 若用户未固定 Hugging Face endpoint,则默认走国内镜像,让本地模型下载
# (embedding / reranker)在内网环境下也能工作.如需覆盖请自行设置
# HF_ENDPOINT.
if not os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class Settings(BaseSettings):
    """运行时配置.所有值都可以通过环境变量覆盖."""

    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), extra="ignore")

    # --- LLM(任何 OpenAI 兼容服务:DeepSeek / 智谱 BigModel 等)----------
    # 代码默认值保持智谱 GLM-5.2(对齐架构文档的锁定选型);实际调用哪家由
    # .env 覆盖 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL 决定.
    llm_api_key: str = ""
    llm_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    llm_model: str = "glm-5.2"

    # --- Embedding --------------------------------------------------------
    embedding_model: str = "embedding-3"
    # Embedding 后端:"zhipu"(OpenAI 兼容 API,需要单独购买 embedding
    # 资源包)或 "local"(sentence-transformers,完全离线).
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "zhipu")
    # 当 EMBEDDING_BACKEND=local 时使用的本地 embedding 模型.
    local_embedding_model: str = os.getenv(
        "LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"
    )
    # 智谱专用的 key 与 endpoint,只服务 "zhipu" embedding 后端.它与上面的
    # LLM 配置**相互独立**:LLM 换到别家后 embedding 仍要连智谱(DeepSeek
    # 不提供 embedding 接口),因此这两个不能被合并.
    zhipu_api_key: str = os.getenv("ZHIPU_API_KEY", "")
    zhipu_embedding_base_url: str = "https://open.bigmodel.cn/api/paas/v4"

    @model_validator(mode="after")
    def _inherit_llm_key(self) -> "Settings":
        """未显式配置 LLM_API_KEY 时,从 ZHIPU_API_KEY 继承.

        只为兼容本适配器之前的老 .env(那时只有一个 key,且 endpoint 就是
        智谱).限定在 endpoint 仍是智谱的情况下继承,避免把智谱的 key 误发
        给别家服务 —— 那样只会换来一个令人费解的 401.
        """
        if not self.llm_api_key and "bigmodel.cn" in self.llm_base_url:
            self.llm_api_key = self.zhipu_api_key
        return self

    # --- 存储路径 ----------------------------------------------------
    kb_dir: Path = BASE_DIR / "data" / "kb"
    notes_dir: Path = BASE_DIR / "data" / "notes"
    chroma_db_dir: Path = BASE_DIR / "data" / "chroma"
    chroma_collection_name: str = "personal_knowledge"
    sqlite_checkpoint_path: Path = BASE_DIR / "data" / "checkpoints.sqlite"

    # --- RAG 调优 -------------------------------------------------------
    top_k: int = 10
    rerank_top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 64
    embedding_batch_size: int = 20

    # --- Reranker ---------------------------------------------------------
    reranker_enabled: bool = True
    reranker_model: str = "bge-reranker-v2-m3"
    reranker_device: str = "cpu"  # "cpu" | "mps" | "cuda"

    # --- 图行为 --------------------------------------------------
    max_history_messages: int = 8
    max_retry: int = 1
    # 单个生成步骤内 model→tool→model 往返轮次的上限.
    max_tool_iterations: int = 3
    # 预留,当前未生效:`OpenAICompatLLM` 支持 `stream_tokens` callback,但
    # 没有地方传入 — `main.py` 不会把它转发给 `build_graph`,所以 CLI 仍然
    # 在结尾一次性打印完整回答.
    stream_tokens: bool = True

    # --- API 错误码(GLM 专用) -------------------------------------
    err_code_insufficient_balance: int = 1301
    err_code_rate_limit: int = 1305

    @property
    def resolve_model_name(self) -> str:
        """发送给 API 的 model id.做成 property 是为了让单处覆盖
        (LLM_MODEL 环境变量)驱动每一次调用."""
        return self.llm_model

    @property
    def _resolved(self) -> dict:
        # 供下方 __post_init__ 风格的校验使用的 helper
        return {
            "kb_dir": self.kb_dir,
            "notes_dir": self.notes_dir,
            "chroma_db_dir": self.chroma_db_dir,
        }

    def ensure_dirs(self) -> None:
        """创建本 agent 需要的所有磁盘目录."""
        for path in (self.kb_dir, self.notes_dir, self.chroma_db_dir,
                     self.sqlite_checkpoint_path.parent):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回(带缓存的)全局 settings 实例."""
    return Settings()


def validate_api_key(settings: Settings) -> None:
    """若缺少 API key,则给出清晰信息并快速失败."""
    if not settings.llm_api_key:
        raise RuntimeError(
            "LLM_API_KEY is not set. Copy .env.example to .env and fill in "
            "your key — DeepSeek: https://platform.deepseek.com, "
            "Zhipu: https://open.bigmodel.cn."
        )


def _device_or_default(device: str | None) -> Literal["cpu", "mps", "cuda"]:
    """规范化 reranker device 字符串."""
    if device and device.lower() in {"cpu", "mps", "cuda"}:
        return device.lower()  # type: ignore[return-value]
    return "cpu"


__all__ = ["Settings", "get_settings", "validate_api_key", "BASE_DIR"]
