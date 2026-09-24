"""Lightweight LLM adapter wrapping the official `zai-sdk` (ZhipuAiClient).

The graph nodes call `llm.ainvoke(messages, tools=None)` where `messages` is a
list of langchain message objects (HumanMessage / SystemMessage / AIMessage /
ToolMessage). This adapter converts those to the `zai` SDK's plain-dict format
and returns a thin result object exposing `.content` and `.tool_calls`, so the
rest of the graph (`_as_text` in nodes.py) works unchanged.

Why not `ChatOpenAI` / `langchain_openai`?

    GLM-5.2 runs its internal thinking by default. With thinking enabled,
    the API returns the reasoning trace in `reasoning_content` and the final
    answer in `content`. If `max_tokens` is too small the thinking trace
    consumes the whole budget and `content` comes back empty — exactly the
    empty-answer bug we saw. The official SDK gives us first-class `thinking`
    control plus a sane token budget, so answers are always generated.

Native function calling

    `tools` accepts langchain `BaseTool` objects (see `tools/`); they are
    serialised to the OpenAI function-calling payload here. The adapter only
    *carries* tool calls — deciding to run them and feeding the results back is
    the caller's job (`nodes._tool_enabled_completion`). Passing no tools
    leaves every call byte-identical to the tool-less implementation.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Iterable

from zai import ZhipuAiClient

logger = logging.getLogger(__name__)

# GLM-5.2 thinking traces can be long; budget plenty of room so the final
# answer is always generated after the reasoning pass.
DEFAULT_MAX_TOKENS = 8192


class ZhipuLLM:
    """Adapter exposing `ainvoke(messages, tools?) -> .content / .tool_calls`."""

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
                           calls stream instead of one-shot create. Streaming
                           is skipped for tool-enabled calls (deltas carry no
                           tool calls).
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.stream_tokens = stream_tokens
        self._client = ZhipuAiClient(api_key=api_key, base_url=base_url)

    # -- public API ----------------------------------------------------------
    async def ainvoke(self, messages: Iterable[Any], tools: Any = None) -> Any:
        """Async-call the model with langchain messages; returns a result object.

        Runs the synchronous SDK call in a thread so async graph nodes don't
        block the event loop.

        Args:
            messages: langchain message objects.
            tools: Optional list of langchain `BaseTool` to expose to the model.
        """
        payload = [self._to_dict(m) for m in messages]
        tool_payload = _tools_to_payload(tools)
        return await asyncio.to_thread(self._invoke, payload, tool_payload)

    def invoke(self, messages: Iterable[Any], tools: Any = None) -> Any:
        """Synchronous variant of `ainvoke`."""
        return self._invoke(
            [self._to_dict(m) for m in messages], _tools_to_payload(tools)
        )

    # -- internals ------------------------------------------------------------
    def _invoke(self, messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None = None) -> "_ZhipuResult":
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if self.thinking is not None:
            kwargs["thinking"] = self.thinking
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        # Streaming only accumulates text deltas, so a tool-enabled call must
        # take the one-shot path or the tool calls would be lost.
        if self.stream_tokens is not None and not tools:
            content = self._invoke_stream(kwargs)
            return _ZhipuResult(content)

        response = self._client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        return _ZhipuResult(msg.content or "", _normalize_tool_calls(msg))

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
        elif role == "assistant":
            # An assistant turn that requested tools must be sent back verbatim,
            # otherwise the tool results that follow have nothing to attach to.
            calls = getattr(message, "tool_calls", None)
            if calls:
                d["tool_calls"] = [_to_api_tool_call(c) for c in calls]
        return d


def _tools_to_payload(tools: Any) -> list[dict[str, Any]] | None:
    """Serialise langchain tools into the OpenAI function-calling payload."""
    if not tools:
        return None

    payload: list[dict[str, Any]] = []
    for tool in tools:
        schema = None
        for attr in ("tool_call_schema", "args_schema"):
            candidate = getattr(tool, attr, None)
            if candidate is not None and hasattr(candidate, "model_json_schema"):
                try:
                    schema = candidate.model_json_schema()
                    break
                except Exception:
                    logger.debug("Could not build a JSON schema from %s", attr,
                                 exc_info=True)
        payload.append({
            "type": "function",
            "function": {
                "name": getattr(tool, "name", "") or "",
                "description": getattr(tool, "description", "") or "",
                "parameters": schema or {"type": "object", "properties": {}},
            },
        })
    return payload


def _to_api_tool_call(call: Any) -> dict[str, Any]:
    """Convert a langchain tool-call dict/object into the API's shape."""
    if isinstance(call, dict):
        name, args, call_id = call.get("name"), call.get("args"), call.get("id")
    else:
        name = getattr(call, "name", None)
        args = getattr(call, "args", None)
        call_id = getattr(call, "id", None)
    return {
        "id": call_id or "",
        "type": "function",
        "function": {
            "name": name or "",
            "arguments": json.dumps(args or {}, ensure_ascii=False),
        },
    }


def _normalize_tool_calls(message: Any) -> list[dict[str, Any]]:
    """Normalize SDK tool calls into `{"id", "name", "args", "parse_error"}`.

    Unparseable `arguments` become `args=None` plus a `parse_error` message
    rather than an empty dict: the caller must tell the model its call was
    malformed instead of silently invoking the tool with no arguments (which
    for `create_note` would mean "saving" an empty note).
    """
    calls: list[dict[str, Any]] = []
    for raw in (getattr(message, "tool_calls", None) or []):
        fn = getattr(raw, "function", None)
        raw_args = getattr(fn, "arguments", None)
        args: Any = None
        parse_error: str | None = None
        if isinstance(raw_args, dict):
            args = raw_args
        else:
            try:
                args = json.loads(raw_args or "{}")
            except (TypeError, ValueError) as exc:
                parse_error = f"参数不是合法 JSON ({exc}): {raw_args!r}"
        calls.append({
            "id": getattr(raw, "id", "") or "",
            "name": getattr(fn, "name", "") or "",
            "args": args,
            "parse_error": parse_error,
        })
    return calls


class _ZhipuResult:
    """Minimal result object exposing `.content`, `.text` and `.tool_calls`."""

    def __init__(self, content: str, tool_calls: list[dict[str, Any]] | None = None) -> None:
        self.content = content
        self.text = content
        self.tool_calls = tool_calls or []

    def __str__(self) -> str:
        return self.content


__all__ = ["ZhipuLLM", "DEFAULT_MAX_TOKENS"]
