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

import logging
from typing import Any, Awaitable, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from config import Settings, get_settings
from graph.edges import BRANCH_DIRECT, BRANCH_KNOWLEDGE
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
        return {"current_query": f"INTENT:{BRANCH_KNOWLEDGE}", "retry_count": 0}

    try:
        label = await _classify_intent(query)
    except Exception as exc:
        logger.warning("Intent classification failed (%s); defaulting to knowledge.", exc)
        label = BRANCH_KNOWLEDGE

    return {"current_query": f"INTENT:{label}", "retry_count": 0}


async def knowledge_search(state: KnowledgeState) -> dict:
    """Execute the knowledge_search tool and store its hits in state.

    The tool call + result are appended to `messages` (as AIMessage/ToolMessage)
    so the generation node sees the standard tool-call transcript.
    """
    query = _latest_user_text(state)
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
    """Chat branch: answer the user directly without retrieval."""
    query = _latest_user_text(state)
    messages = _trim_history(state) + [HumanMessage(content=query)]
    sys = SystemMessage(content=DIRECT_RESPONSE_SYSTEM_PROMPT)

    try:
        response = await DEPS.llm.ainvoke([sys] + messages)
        text = _as_text(response)
    except Exception as exc:
        logger.exception("direct_response failed")
        text = f"抱歉，我暂时无法处理。原因: {exc}"

    return {"final_answer": text, "messages": [AIMessage(content=text)]}


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
    query = state.get("current_query", "")
    if query.startswith("INTENT:"):
        query = query.split(":", 1)[1] if ":" in query else ""

    context = _render_context(hits)
    messages = _trim_history(state) + [
        HumanMessage(content=f"用户问题: {query}"),
    ]

    sys = SystemMessage(content=_generate_system_prompt(context))
    try:
        response = await DEPS.llm.ainvoke([sys] + messages)
        text = _as_text(response)
    except Exception as exc:
        logger.exception("generate_node failed")
        text = f"抱歉，生成回答时出现错误: {exc}"

    return {"final_answer": text, "messages": [AIMessage(content=text)]}


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


def _trim_history(state: KnowledgeState, settings: Settings | None = None) -> list:
    """Keep the last N human/ai turns (budgeted) for context."""
    settings = settings or get_settings()
    budget = settings.max_history_messages
    messages = state.get("messages", [])
    # Exclude ToolMessages (transient per-turn), keep conversation turns.
    conversational = [m for m in messages if not isinstance(m, ToolMessage)]
    return conversational[-budget:]


def _as_text(response: Any) -> str:
    if isinstance(response, AIMessage):
        return response.content or ""
    if isinstance(response, str):
        return response
    if hasattr(response, "content"):
        return str(response.content or "")
    return str(response)


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
