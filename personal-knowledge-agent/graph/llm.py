"""轻量 LLM 适配器,包装任意 OpenAI 兼容的 chat completions 服务.

graph node 通过 `llm.ainvoke(messages, tools=None)` 调用,其中 `messages` 是
langchain 消息对象列表(HumanMessage / SystemMessage / AIMessage /
ToolMessage).本适配器把它们转成普通 dict,并把响应归一化成一个只暴露
`.content` / `.text` / `.tool_calls` 的轻量结果对象,因此 graph 其余部分
(nodes.py 中的 `_as_text`)无需改动即可工作.

为什么需要这一层?

    它把「用哪家模型」收敛成三行配置(base_url / model / api_key).DeepSeek 与
    智谱 BigModel 都提供 OpenAI 兼容端点,因此同一份适配器可以服务两者 —— 换厂商
    不需要碰任何节点代码.

thinking 与 reasoning_content

    带思考能力的模型(DeepSeek 的 thinking 模式、GLM-5.2 的默认行为)会把推理
    轨迹与最终回答分开返回:推理在 `reasoning_content`,回答在 `content`.两个后果:

    1. `max_tokens` 给得太小,推理轨迹会吃掉全部额度,`content` 返回**空字符串**.
    2. **请求带 `tools=` 时,历史轮次的 `reasoning_content` 必须回传**,否则 API
       会拒绝该请求(不带 `tools=` 时则会被忽略).这是 DeepSeek 的明确约定,
       GLM 也接受该字段.

    为此适配器把 `reasoning_content` 一并取出,节点侧存进
    `AIMessage.additional_kwargs`,再由 `_to_dict` 原样回传.

    `thinking` 这类非标准参数必须经 `extra_body` 下发:openai SDK 只把它序列化成
    请求体的顶层字段,这恰好是各家期望的形状.

原生 Function Calling

    `tools` 接受 langchain `BaseTool` 对象(见 `tools/`);它们在这里被序列化成
    OpenAI function-calling 载荷.适配器只 *携带* tool calls —— 决定是否执行并把
    结果回灌是调用方的职责(`nodes._tool_enabled_completion`).不传 tools 时,
    调用参数与无工具实现完全一致.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# 带思考的模型推理轨迹可能很长;留足空间,确保推理阶段之后总能生成最终回答.
DEFAULT_MAX_TOKENS = 8192


class OpenAICompatLLM:
    """暴露 `ainvoke(messages, tools?) -> .content / .tool_calls` 的适配器."""

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
        """创建一个适配器.

        Args:
            api_key: 服务商 API key.
            model: 模型 id(默认 glm-5.2;DeepSeek 用 deepseek-flash 等).
            base_url: 服务的 OpenAI 兼容 endpoint.
            temperature: 采样温度.注意带思考的模型会**静默忽略**该参数.
            max_tokens: 最大输出 token 数(默认 8192,足够装下推理轨迹 + 最终回答).
            thinking: 例如 {"type": "disabled"} 可为廉价的分类调用关闭思考;
                      None 则保持模型默认(DeepSeek 与 GLM 都默认开启).
            stream_tokens: 可选回调,每收到一个流式 token 块即调用(供交互式 CLI
                           使用).提供时改用流式而非一次性 create.启用了工具的
                           调用会跳过流式(delta 不携带 tool calls).
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.stream_tokens = stream_tokens
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    # -- 公开 API ------------------------------------------------------------
    async def ainvoke(self, messages: Iterable[Any], tools: Any = None) -> Any:
        """用 langchain 消息 async 调用模型;返回结果对象.

        `AsyncOpenAI` 本身是异步的,因此不再需要把同步 SDK 丢进线程池.

        Args:
            messages: langchain 消息对象.
            tools: 可选,暴露给模型的 langchain `BaseTool` 列表.
        """
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=[self._to_dict(m) for m in messages],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        tool_payload = _tools_to_payload(tools)
        if self.thinking is not None:
            # 非标准参数只能经 extra_body 下发(见模块 docstring).
            kwargs["extra_body"] = {"thinking": self.thinking}
        if tool_payload:
            kwargs["tools"] = tool_payload
            kwargs["tool_choice"] = "auto"

        # 流式只累积文本 delta,因此启用了工具的调用必须走一次性路径,
        # 否则 tool calls 会丢失.
        if self.stream_tokens is not None and not tool_payload:
            return _LLMResult(await self._invoke_stream(kwargs))

        response = await self._client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        return _LLMResult(
            msg.content or "",
            _normalize_tool_calls(msg),
            # `reasoning_content` 是服务商扩展字段,非标准响应里不存在.
            getattr(msg, "reasoning_content", None),
        )

    # -- 内部实现 -------------------------------------------------------------
    async def _invoke_stream(self, kwargs: dict[str, Any]) -> str:
        """流式获取响应,并把 token 转发给 `self.stream_tokens`."""
        chunks: list[str] = []
        stream = await self._client.chat.completions.create(**kwargs, stream=True)
        async for chunk in stream:
            if not chunk.choices:
                continue
            token = getattr(chunk.choices[0].delta, "content", None) or ""
            if token:
                chunks.append(token)
                try:
                    self.stream_tokens(token)
                except Exception:
                    logger.debug("stream_tokens callback failed", exc_info=True)
        return "".join(chunks)

    @staticmethod
    def _to_dict(message: Any) -> dict[str, Any]:
        """把 langchain 消息转成 chat completions 的普通 dict 格式."""
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
            # 请求过工具的 assistant 轮次必须原样回传,
            # 否则紧随其后的工具结果将无处挂载.
            calls = getattr(message, "tool_calls", None)
            if calls:
                d["tool_calls"] = [_to_api_tool_call(c) for c in calls]
            # 带思考的模型:请求带 tools= 时,历史轮次的推理轨迹必须一起回传,
            # 否则 API 直接拒绝.不带 tools= 的请求里该字段会被忽略,因此只要
            # 手上有就带上,无需判断当前这次调用是否带工具.
            reasoning = (getattr(message, "additional_kwargs", None) or {}).get(
                "reasoning_content"
            )
            if reasoning:
                d["reasoning_content"] = reasoning
        return d


def _tools_to_payload(tools: Any) -> list[dict[str, Any]] | None:
    """把 langchain tools 序列化成 OpenAI function-calling 载荷."""
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
    """把 langchain 的 tool-call dict/对象转成 API 的形状."""
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
    """把 SDK tool calls 归一化成 `{"id", "name", "args", "parse_error"}`.

    无法解析的 `arguments` 会变成 `args=None` 外加一条 `parse_error` 信息,
    而不是空 dict:调用方必须告诉模型它的调用格式不对,而不是静默地以无参数
    方式执行工具(对 `create_note` 而言,那等于"保存"一条空笔记).
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


class _LLMResult:
    """最小结果对象,暴露 `.content`,`.text`,`.tool_calls` 与 `.reasoning_content`."""

    def __init__(
        self,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
    ) -> None:
        self.content = content
        self.text = content
        self.tool_calls = tool_calls or []
        # 调用方(`nodes._ai_message_with_calls`)需要把它存进 assistant 轮次,
        # 以便带 tools= 的后续请求能原样回传.
        self.reasoning_content = reasoning_content or ""

    def __str__(self) -> str:
        return self.content


__all__ = ["OpenAICompatLLM", "DEFAULT_MAX_TOKENS"]
