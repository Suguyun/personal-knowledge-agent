"""Lightweight LLM adapter wrapping the official `zai-sdk` (ZhipuAiClient).

The graph nodes call `llm.ainvoke(messages)` where `messages` is a list of
langchain message objects (HumanMessage / SystemMessage / AIMessage /
ToolMessage). This adapter converts those to the `zai` SDK's plain-dict
format and returns a thin result object exposing `.content`, so the rest of
the graph (`_as_text` in nodes.py) works unchanged.

Why not `ChatOpenAI` / `langchain_openai`?

    GLM-5.2 runs its internal thinking by default. With thinking enabled,
    the API returns the reasoning trace in `reasoning_content` and the final
    answer in `content`. If `max_tokens` is too small the thinking trace
    consumes the whole budget and `content` comes back empty — exactly the
    empty-answer bug we saw. The official SDK gives us first-class `thinking`
    control plus a sane token budget, so answers are always generated.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Iterable

from zai import ZhipuAiClient

logger = logging.getLogger(__name__)

# GLM-5.2 thinking traces can be long; budget plenty of room so the final
# answer is always generated after the reasoning pass.
DEFAULT_MAX_TOKENS = 8192


class ZhipuLLM:
    """Adapter exposing `ainvoke(messages) -> .content` over zai-sdk."""

    def __init__(
        self,
        api_key: str,
        model: str = "glm-5.2",
        base_url: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        thinking: dict[str, Any] | None = None,
        stream_tokens: Any | None = None,
    ) -> None:
        """Create a ZhipuLLM client.

        Args:
            api_key: Zhipu API key.
            model: GLM model id (default glm-5.2).
            base_url: Optional custom base URL (defaults to SDK's own).
            temperature: Sampling temperature.
            max_tokens: Max output tokens (default 8192, enough for the
                        thinking trace + final answer).
            thinking: e.g. {"type": "disabled"} to turn off thinking for
                      cheap classification calls; None keeps the model default.
            stream_tokens: Optional callback invoked with each streamed token
                           chunk (used by the interactive CLI). When provided,
                           calls stream instead of one-shot create.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.stream_tokens = stream_tokens
        self._client = ZhipuAiClient(api_key=api_key, base_url=base_url)

    # -- public API ----------------------------------------------------------
    async def ainvoke(self, messages: Iterable[Any]) -> Any:
        """Async-call the model with langchain messages; returns a .content object.

        Runs the synchronous SDK call in a thread so async graph nodes don't
        block the event loop.
        """
        payload = [self._to_dict(m) for m in messages]
        return await asyncio.to_thread(self._invoke, payload)

    def invoke(self, messages: Iterable[Any]) -> Any:
        """Synchronous variant of `ainvoke`."""
        return self._invoke([self._to_dict(m) for m in messages])

    # -- internals ------------------------------------------------------------
    def _invoke(self, messages: list[dict[str, Any]]) -> "_ZhipuResult":
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if self.thinking is not None:
            kwargs["thinking"] = self.thinking

        if self.stream_tokens is not None:
            content = self._invoke_stream(kwargs)
            return _ZhipuResult(content)

        response = self._client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        return _ZhipuResult(msg.content or "")

    def _invoke_stream(self, kwargs: dict[str, Any]) -> str:
        """Stream the response, forwarding tokens to `self.stream_tokens`."""
        chunks: list[str] = []
        stream = self._client.chat.completions.create(**kwargs, stream=True)
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            token = getattr(delta, "content", None) or ""
            if token:
                chunks.append(token)
                try:
                    self.stream_tokens(token)
                except Exception:
                    logger.debug("stream_tokens callback failed", exc_info=True)
        return "".join(chunks)

    @staticmethod
    def _to_dict(message: Any) -> dict[str, Any]:
        """Convert a langchain message to the zai SDK plain-dict format."""
        role_map = {
            "HumanMessage": "user",
            "SystemMessage": "system",
            "AIMessage": "assistant",
            "ToolMessage": "tool",
        }
        cls = type(message).__name__
        role = role_map.get(cls, "user")
        content = getattr(message, "content", "") or ""

        d: dict[str, Any] = {"role": role, "content": content}
        if role == "tool":
            d["tool_call_id"] = getattr(message, "tool_call_id", None)
        return d


class _ZhipuResult:
    """Minimal result object exposing `.content` (and `.text` for safety)."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.text = content

    def __str__(self) -> str:
        return self.content


__all__ = ["ZhipuLLM", "DEFAULT_MAX_TOKENS"]
