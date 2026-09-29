"""Embedding 后端.

支持两种后端,由 `EMBEDDING_BACKEND` 选择:

- `zhipu`(默认):通过 OpenAI 兼容接口调用智谱 `embedding-3`,
  按 `embedding_batch_size` 分组做批量请求.账号需要购买 embedding 资源包.
- `local`:在本地加载 `sentence-transformers` 模型(如
  `BAAI/bge-small-zh-v1.5`).完全离线,不占 API 配额,零成本.适合在
  智谱账号有 chat 配额但没有 embedding 资源包时快速起步.

`Embedder` 持有惰性初始化的后端 client,并暴露轻量的 `embed_texts`
接口,使调用方(index 构建 + 查询链路)无需接触后端细节.
"""

from __future__ import annotations

import logging

from config import Settings, get_settings

logger = logging.getLogger(__name__)


class Embedder:
    """可切换后端的 embedding client(zhipu API 或本地模型)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._zhipu_client: object | None = None
        self._local_model: object | None = None

    @property
    def backend(self) -> str:
        """归一化后的后端名称(zhipu | local)."""
        return (self.settings.embedding_backend or "zhipu").strip().lower()

    @property
    def client(self):
        """惰性初始化的智谱 OpenAI 兼容 client(zhipu 后端)."""
        if self._zhipu_client is None:
            from openai import OpenAI

            self._zhipu_client = OpenAI(
                api_key=self.settings.zhipu_api_key,
                base_url=self.settings.openai_base_url,
            )
        return self._zhipu_client

    @property
    def local_model(self):
        """惰性加载的 sentence-transformers 模型(local 后端)."""
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading local embedding model %s …",
                        self.settings.local_embedding_model)
            self._local_model = SentenceTransformer(self.settings.local_embedding_model)
        return self._local_model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """对一组文本做 embedding;按顺序为每个输入返回一个 float 向量."""
        if not texts:
            return []

        if self.backend == "local":
            return self._embed_local(texts)

        return self._embed_zhipu(texts)

    # --- zhipu 后端 -----------------------------------------------------------
    def _embed_zhipu(self, texts: list[str]) -> list[list[float]]:
        batch_size = self.settings.embedding_batch_size
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(
                model=self.settings.embedding_model,
                input=batch,
            )
            # API 按请求顺序返回 items;这里按 index 做防御性排序,
            # 使顺序永远不会静默依赖上游行为.
            ordered = sorted(response.data, key=lambda item: item.index)
            results.extend([item.embedding for item in ordered])
        logger.debug("Embedded %d text(s) via zhipu in %d batch(es).",
                     len(texts), (len(texts) + batch_size - 1) // batch_size)
        return results

    # --- local 后端 -----------------------------------------------------------
    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        batch_size = self.settings.embedding_batch_size
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors = self.local_model.encode(batch, normalize_embeddings=True)
            # encode() 返回 numpy 数组;归一化为 python 的 float 列表,
            # 使返回契约在所有后端之间保持一致.
            results.extend([v.tolist() for v in vectors])
        logger.debug("Embedded %d text(s) via local model %s.",
                     len(texts), self.settings.local_embedding_model)
        return results


__all__ = ["Embedder"]
