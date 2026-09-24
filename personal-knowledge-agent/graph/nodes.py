"""Graph nodes.

Node responsibilities:

    intent_router       — cheap, thinking-disabled classifier: knowledge vs direct
    knowledge_search    — run the knowledge_search tool (retrieve + store chunks)
    direct_response     — chat answer without retrieval
    rerank_node         — rerank the retrieved candidates (via the tool's own
                          retriever; kept as an explicit graph step so the
                          state's `retrieved_docs` always reflects reranked
                          order and the LLM context can be pruned)
    generate_node       — synthesize the final answer with citations
    rewrite_query_node  — rewrite a failed query once, then re-retrieve

All nodes are async (spec: "all nodes must be async-compatible") and read
dependencies from a module-level `DEPENDENCIES` holder set by `build_graph`.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Awaitable, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from config import Settings, get_settings
from graph.edges import BRANCH_DIRECT, BRANCH_KNOWLEDGE, INTENT_MARKERS, INTENT_PREFIX
from graph.state import KnowledgeState
from prompts.system_prompt import (
    DIRECT_RESPONSE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dependency injection (filled by build_graph)
# ---------------------------------------------------------------------------
class _Deps:
    settings: Settings | None = None
    llm: Any = None                # ZhipuLLM (glm-5.2)
    tools: list[BaseTool] = []
    tool_by_name: dict[str, BaseTool] = {}
    stream_tokens: Callable[[str], None] | None = None


DEPS = _Deps()


def _set_deps(settings, llm, tools, client, stream_tokens) -> None:
    # `client` is kept for backward-compat with callers but no longer used:
    # all LLM calls go through the ZhipuLLM adapter (`llm`).
    DEPS.settings = settings
    DEPS.llm = llm
    DEPS.tools = tools
    DEPS.tool_by_name = {t.name: t for t in tools}
    DEPS.stream_tokens = stream_tokens


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------
async def intent_router(state: KnowledgeState) -> dict:
    """Classify the latest user message as knowledge or direct.

    Uses a thinking-disabled GLM call (cheap) and stores the decision on
    `current_query` as `INTENT:<branch>`; `route_intent` maps it back.
    """
    query = _latest_user_text(state)
    if not query:
        return {"current_query": f"{INTENT_PREFIX}{BRANCH_KNOWLEDGE}", "retry_count": 0}

    try:
        label = await _classify_intent(query)
    except Exception as exc:
        logger.warning("Intent classification failed (%s); defaulting to knowledge.", exc)
        label = BRANCH_KNOWLEDGE

    return {"current_query": f"{INTENT_PREFIX}{label}", "retry_count": 0}


async def knowledge_search(state: KnowledgeState) -> dict:
    """Execute the knowledge_search tool and store its hits in state.

    The tool call + result are appended to `messages` (as AIMessage/ToolMessage)
    so the generation node sees the standard tool-call transcript.
    """
    query = _active_query(state)
    tool = DEPS.tool_by_name.get("knowledge_search")
    if tool is None:
        return {"retrieved_docs": []}

    try:
        result = await tool.ainvoke({"query": query})
        hits = _parse_hits(result)
    except Exception as exc:
        logger.exception("knowledge_search tool failed")
        result = f"检索失败: {exc}"
        hits = []

    ai_msg = AIMessage(
        content="",
        tool_calls=[{
            "name": tool.name,
            "args": {"query": query},
            "id": f"call_{abs(hash(query)) % 10**8}",
        }],
    )
    tool_msg = ToolMessage(content=str(result), tool_call_id=ai_msg.tool_calls[0]["id"])

    return {
        "messages": [ai_msg, tool_msg],
        "retrieved_docs": hits,
        "current_query": query,
    }


async def direct_response(state: KnowledgeState) -> dict:
    """Chat branch: answer directly, without retrieval, but with tools.

    Tools matter on this branch too: "帮我记一下…" is chat-shaped, not
    knowledge-shaped, so `create_note` has to be reachable here as well.
    """
    query = _latest_user_text(state)
    messages = _trim_history(state) + [HumanMessage(content=query)]
    sys = SystemMessage(content=DIRECT_RESPONSE_SYSTEM_PROMPT)

    try:
        text, transcript = await _tool_enabled_completion([sys] + messages)
    except Exception as exc:
        logger.exception("direct_response failed")
        text, transcript = f"抱歉，我暂时无法处理。原因: {exc}", []

    return {"final_answer": text, "messages": [*transcript, AIMessage(content=text)]}


async def rerank_node(state: KnowledgeState) -> dict:
    """Re-order retrieved_docs (reranker runs inside the retriever/tool).

    Kept as an explicit node so the graph's flow matches the architecture
    diagram (… → rerank_node → generate_node → …) and so we can prune the
    tool message that grows too large before generation.
    """
    hits = state.get("retrieved_docs") or []
    return {"retrieved_docs": hits}


async def generate_node(state: KnowledgeState) -> dict:
    """Synthesize the final, cited answer from the retrieved chunks."""
    hits = state.get("retrieved_docs") or []
    query = _active_query(state)

    context = _render_context(hits)
    messages = _trim_history(state) + [
        HumanMessage(content=f"用户问题: {query}"),
    ]

    sys = SystemMessage(content=_generate_system_prompt(context))
    try:
        text, transcript = await _tool_enabled_completion([sys] + messages)
    except Exception as exc:
        logger.exception("generate_node failed")
        text, transcript = f"抱歉，生成回答时出现错误: {exc}", []

    return {"final_answer": text, "messages": [*transcript, AIMessage(content=text)]}


async def rewrite_query_node(state: KnowledgeState) -> dict:
    """Rewrite the failed query (once) to improve retrieval recall."""
    original = _latest_user_text(state)
    retry_count = int(state.get("retry_count", 0)) + 1

    rewritten = await _rewrite(original, state)
    # Keep the state's query as the *rewritten* query for the retrieval node,
    # but preserve the original user message in `messages` for context.
    return {"current_query": rewritten, "retry_count": retry_count}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _latest_user_text(state: KnowledgeState) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return (msg.content or "").strip()
    return ""


def _active_query(state: KnowledgeState) -> str:
    """The query that should drive this turn's retrieval *and* generation.

    `current_query` holds the real query once the turn is under way, but
    `intent_router` first parks an `INTENT:<branch>` marker there — so only a
    non-marker value counts as a query. This is what makes the retry loop
    work: `rewrite_query_node` writes the rewritten query to `current_query`,
    and retrieval has to pick that up instead of re-reading the original user
    message (which would make the second retrieval identical to the first).

    Compared against the exact marker set, not a prefix test: a rewritten
    query could itself start with `INTENT:` (the user may well be asking about
    intent routing), and treating that as a marker would silently fall back to
    the original query — reintroducing the very bug this guard exists for.
    """
    current = (state.get("current_query") or "").strip()
    if current and current not in INTENT_MARKERS:
        return current
    return _latest_user_text(state)


def _trim_history(state: KnowledgeState, settings: Settings | None = None) -> list:
    """Keep the last N human/ai turns (budgeted) for context."""
    settings = settings or get_settings()
    budget = settings.max_history_messages
    messages = state.get("messages", [])
    # Exclude ToolMessages (transient per-turn) and the content-less AIMessage
    # that `knowledge_search` appends to carry its tool_calls: `ZhipuLLM` drops
    # tool_calls when converting to the SDK payload, so forwarding it would put
    # an empty assistant turn in front of the model. Neither is conversation.
    # The `isinstance(m.content, str)` guard matters: langchain allows
    # AIMessage(content=[{...}]) for tool/multimodal turns, and a list has no
    # .strip() — such a message must be kept, not crash the turn.
    conversational = [
        m
        for m in messages
        if not isinstance(m, ToolMessage)
        and not (
            isinstance(m, AIMessage)
            and isinstance(m.content, str)
            and not m.content.strip()
        )
    ]
    # The graph appends the current turn's HumanMessage to `messages` before a
    # node runs, and every caller re-adds the question itself (raw in
    # direct_response, framed as "用户问题: …" in generate_node). Leaving it in
    # would send the question twice in two framings and burn one of the
    # `max_history_messages` slots on it.
    while conversational and isinstance(conversational[-1], HumanMessage):
        conversational.pop()
    return conversational[-budget:]


def _as_text(response: Any) -> str:
    if isinstance(response, AIMessage):
        return response.content or ""
    if isinstance(response, str):
        return response
    if hasattr(response, "content"):
        return str(response.content or "")
    return str(response)


# ---------------------------------------------------------------------------
# Native function calling (tool loop)
# ---------------------------------------------------------------------------
def _llm_accepts_tools(llm: Any) -> bool:
    """Whether `llm.ainvoke` takes a `tools=` keyword.

    Checked up front rather than by catching TypeError, so an unrelated
    TypeError raised *inside* a call is never mistaken for "no tool support".
    """
    try:
        return "tools" in inspect.signature(llm.ainvoke).parameters
    except (TypeError, ValueError):
        return False


def _response_tool_calls(response: Any) -> list[dict[str, Any]]:
    """Tool calls requested by the model, normalized by the LLM adapter."""
    calls = getattr(response, "tool_calls", None) or []
    return [c for c in calls if isinstance(c, dict)]


def _ai_message_with_calls(response: Any, calls: list[dict[str, Any]]) -> AIMessage:
    """Rebuild the assistant turn that requested tools (for the transcript)."""
    return AIMessage(
        content=_as_text(response),
        tool_calls=[
            {
                "name": c.get("name") or "",
                "args": c.get("args") if isinstance(c.get("args"), dict) else {},
                "id": c.get("id") or "",
            }
            for c in calls
        ],
    )


async def _execute_tool_call(call: dict[str, Any]) -> str:
    """Run one model-requested tool call and return its result as text.

    Per spec, tool failures come back to the model as structured text so it can
    decide what to do next, instead of aborting the run.
    """
    name = (call.get("name") or "").strip()
    if call.get("parse_error"):
        return f"工具调用参数解析失败: {call['parse_error']}"
    tool = DEPS.tool_by_name.get(name)
    if tool is None:
        return (f"错误: 不存在名为 {name!r} 的工具。"
                f"可用工具: {sorted(DEPS.tool_by_name)}")
    args = call.get("args")
    if not isinstance(args, dict):
        return f"错误: {name} 的参数必须是 JSON 对象，实际收到 {type(args).__name__}。"
    try:
        return str(await tool.ainvoke(args))
    except Exception as exc:
        logger.exception("Tool %s failed", name)
        return f"工具 {name} 执行失败: {exc}"


async def _tool_enabled_completion(conversation: list) -> tuple[str, list]:
    """Call the model, running whatever tools it asks for, until it answers.

    When the model requests no tool — the case for every query that doesn't
    need one — this is a single call with the same arguments as the tool-less
    implementation, so existing behaviour is unchanged.

    Args:
        conversation: system + history + current-turn messages.

    Returns:
        `(final_text, transcript)`, where transcript holds the extra
        AIMessage/ToolMessage pairs (chronological) to append to the state.
    """
    settings = DEPS.settings or get_settings()
    budget = max(1, int(settings.max_tool_iterations))
    tools = DEPS.tools or None
    use_tools = bool(tools) and DEPS.llm is not None and _llm_accepts_tools(DEPS.llm)

    convo = list(conversation)
    transcript: list = []

    for iteration in range(budget):
        if use_tools:
            try:
                response = await DEPS.llm.ainvoke(convo, tools=tools)
            except Exception:
                # If the API rejects our tools payload, keep the agent usable by
                # falling back to the plain call rather than failing every query.
                logger.warning("Tool-enabled call failed; retrying without tools.",
                               exc_info=True)
                use_tools = False
                response = await DEPS.llm.ainvoke(convo)
        else:
            response = await DEPS.llm.ainvoke(convo)

        calls = _response_tool_calls(response)
        if not calls:
            return _as_text(response), transcript

        ai_msg = _ai_message_with_calls(response, calls)
        convo.append(ai_msg)
        transcript.append(ai_msg)
        for call in calls:
            tool_msg = ToolMessage(
                content=await _execute_tool_call(call),
                tool_call_id=call.get("id") or "",
            )
            convo.append(tool_msg)
            transcript.append(tool_msg)
        logger.info("Tool iteration %d/%d: ran %d call(s).",
                    iteration + 1, budget, len(calls))

    logger.warning("Tool loop hit its %d-iteration cap.", budget)
    return "（已达到工具调用次数上限，未能给出最终回答。）", transcript


def _parse_hits(raw: Any) -> list[dict[str, Any]]:
    """Parse the tool's JSON string result back into a hit list."""
    import json

    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _render_context(hits: list[dict[str, Any]]) -> str:
    """Render retrieved chunks into the generation prompt's context block."""
    if not hits:
        return "(本次检索未返回任何知识库内容)"

    blocks = []
    for i, hit in enumerate(hits, start=1):
        meta = hit.get("metadata") or {}
        source = meta.get("source_doc", "未知文档")
        section = meta.get("section_header", "")
        loc = f"{source} · {section}" if section else source
        blocks.append(f"[{i}] 来源: {loc}\n{hit.get('content', '')}")
    return "\n\n".join(blocks)


def _generate_system_prompt(context: str) -> str:
    """System prompt for the generation step, with the context injected."""
    return SYSTEM_PROMPT + f"""

## 本次检索到的知识库内容

以下是针对当前问题的知识库检索结果。请只依据这些内容回答，并按要求标注来源。
如果检索结果与问题无关或为空，请直接说明知识库中未找到相关内容。

--- 检索结果开始 ---
{context}
--- 检索结果结束 ---
"""


async def _classify_intent(query: str) -> str:
    """Thinking-disabled GLM call: knowledge vs direct (no retrieval)."""
    system = (
        "判断下面这句用户消息是否需要查询个人知识库。"
        "需要检索知识（关于用户个人记录、资料、笔记、文档、过往内容等）返回 knowledge；"
        "仅是寒暄/闲聊/询问助手能力/与知识库无关的开放性问题返回 direct。"
        "只输出 knowledge 或 direct，不要输出其他内容。"
    )
    if DEPS.settings is None or not DEPS.settings.zhipu_api_key:
        raise RuntimeError("未配置 ZHIPU_API_KEY，无法进行分类路由。")
    # Thinking disabled + tiny token budget keeps routing cheap and fast.
    classifier = _make_classifier_llm(DEPS.settings)
    response = await classifier.ainvoke(
        [SystemMessage(content=system), HumanMessage(content=query)]
    )
    label = (response.content or "").strip().lower()
    if label in {BRANCH_KNOWLEDGE, BRANCH_DIRECT}:
        return label
    return BRANCH_KNOWLEDGE


def _make_classifier_llm(settings: Settings) -> Any:
    """Build a ZhipuLLM tuned for cheap, thinking-disabled classification."""
    from graph.llm import ZhipuLLM

    return ZhipuLLM(
        api_key=settings.zhipu_api_key,
        model=settings.resolve_model_name,
        base_url=settings.openai_base_url,
        temperature=0,
        max_tokens=16,
        thinking={"type": "disabled"},
    )


async def _rewrite(query: str, state: KnowledgeState) -> str:
    """Ask GLM to rewrite a query that failed to retrieve useful content."""
    messages = _trim_history(state)
    prompt = (
        f"上一条查询未能从知识库中检索到有用信息。请将下面的查询改写得更具体、"
        f"更可能命中个人笔记/文档，只输出改写后的查询本身，不要解释。\n\n原查询: {query}"
    )
    try:
        sys = SystemMessage(content="你是一名检索查询改写助手，输出简洁。")
        response = await DEPS.llm.ainvoke([sys] + messages + [HumanMessage(content=prompt)])
        rewritten = _as_text(response).strip().strip('"')
        return rewritten or query
    except Exception as exc:
        logger.warning("Query rewrite failed (%s); using original.", exc)
        return query


__all__ = [
    "intent_router",
    "knowledge_search",
    "direct_response",
    "rerank_node",
    "generate_node",
    "rewrite_query_node",
    "_set_deps",
]
