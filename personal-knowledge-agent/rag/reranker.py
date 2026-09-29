"""对 top-k 检索结果做可选 cross-encoder rerank.

通过 `sentence-transformers` 的 CrossEncoder 使用 `bge-reranker-v2-m3`.
模型首次使用时下载(约 2GB).加载在第一次调用 `rerank` 时惰性进行,
因此从不检索(或关闭了 rerank)的应用不会付出启动开销.

降级契约(按规格):若 reranker 不可用或执行失败,调用方(`Retriever`)
回退到原始检索分数.本模块只在构造期抛出异常;`rerank` 在内部捕获运行时
失败.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from config import Settings, get_settings

logger = logging.getLogger(__name__)


class Reranker:
    """惰性加载模型的 cross-encoder reranker."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model: Any | None = None
        self._load_error: str | None = None

    def _ensure_model(self) -> Any:
        """只加载一次 cross-encoder;无法加载时抛出明确错误."""
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            raise RuntimeError(self._load_error)

        if not self.settings.reranker_enabled:
            raise RuntimeError("Reranker disabled via RERANKER_ENABLED=false.")

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            self._load_error = (
                "sentence-transformers not installed; run "
                "`pip install sentence-transformers` or set "
                "RERANKER_ENABLED=false."
            )
            raise RuntimeError(self._load_error) from exc

        # bge-reranker-v2-m3 训练时带有 query/passage 指令前缀;在 zh/en 混合
        # 内容上应用它可提升排序准确率.`max_length` 会截断过长的 chunk,
        # 使模型能接受所有输入.
        self._model = CrossEncoder(
            model_name=self.settings.reranker_model,
            device=self.settings.reranker_device,
            max_length=512,
        )
        logger.info("Loaded reranker %s on %s.",
                    self.settings.reranker_model, self.settings.reranker_device)
        return self._model

    def rerank(
        self,
        query: str,
        hits: list[dict[str, Any]],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """用 `query` 对 `hits`(已包含 `content`)做 rerank.

        Args:
            query: 用户查询(或改写后的查询).
            hits: 来自 vector store 的候选 chunk,每个都含 `content`.
            top_n: rerank 后保留的结果数量.

        Returns:
            重排后的 `hits`(原始 dict,score 更新为 sigmoid 后的
            cross-encoder 分数)截断到 `top_n`.任何失败都返回空列表 ——
            此时调用方降级为原始检索.
        """
        if not hits:
            return []

        try:
            model = self._ensure_model()
        except Exception as exc:
            logger.warning("Reranker unavailable (%s); using raw retrieval.", exc)
            return []

        try:
            pairs = [
                [_PREFIX + query, hit["content"][:512]] for hit in hits
            ]
            scores = model.predict(pairs, show_progress_bar=False)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Rerank failed (%s); using raw retrieval.", exc)
            return []

        scored = list(zip(hits, scores))
        scored.sort(key=lambda item: float(item[1]), reverse=True)
        # Cross-encoder 分数是原始 logits(约 ±5);压缩到 [0,1],使下游
        # 的 score 处理与向量相似度保持一致.
        top = scored[:top_n]
        return [
            {**hit, "score": round(1.0 / (1.0 + math.exp(-float(s))), 4)}
            for hit, s in top
        ]


_PREFIX = "为这个句子生成表示以用于检索相关文章："


__all__ = ["Reranker"]
