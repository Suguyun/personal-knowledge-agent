"""Graph nodes.

各 node 职责:

    intent_router       — 廉价,关闭 thinking 的分类器:knowledge 还是 direct
    knowledge_search    — 执行 knowledge_search 工具(检索 + 存储文本块)
    direct_response     — 不经检索的聊天回答
    rerank_node         — 对检索到的候选做 rerank(借工具自带的 retriever
                          完成;保留为显式的 graph 步骤,好让 state 的
                          `retrieved_docs` 始终反映 rerank 后的顺序,
                          也便于裁剪 LLM 上下文)
    generate_node       — 带引用地合成最终回答
    rewrite_query_node  — 把失败的查询改写一次,然后重新检索

所有 node 都是 async(规格:"all nodes must be async-compatible"),并从由
`build_graph` 设置的模块级 `DEPENDENCIES` 容器读取依赖.
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
# 依赖注入(由 build_graph 填充)
# ---------------------------------------------------------------------------
class _Deps:
    settings: Settings | None = None
    llm: Any = None                # ZhipuLLM(glm-5.2)
    tools: list[BaseTool] = []
    tool_by_name: dict[str, BaseTool] = {}
    stream_tokens: Callable[[str], None] | None = None


DEPS = _Deps()


def _set_deps(settings, llm, tools, client, stream_tokens) -> None:
    # `client` 为兼容调用方保留,但已不再使用:
    # 所有 LLM 调用都走 ZhipuLLM 适配器(`llm`).
    DEPS.settings = settings
    DEPS.llm = llm
    DEPS.tools = tools
    DEPS.tool_by_name = {t.name: t for t in tools}
    DEPS.stream_tokens = stream_tokens


# ---------------------------------------------------------------------------
# Node 实现
# ---------------------------------------------------------------------------
async def intent_router(state: KnowledgeState) -> dict:
    """把最新的用户消息分类为 knowledge 或 direct.

    使用关闭 thinking 的 GLM 调用(廉价),并把决策以 `INTENT:<branch>`
    存到 `current_query`;`route_intent` 再映射回来.
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
    """执行 knowledge_search 工具,并把命中结果存入 state.

    工具调用 + 结果会(以 AIMessage/ToolMessage 形式)追加到 `messages`,
    好让 generate node 看到标准的 tool-call 记录.
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
    """Chat 分支:不检索直接回答,但要带 tools.

    这条分支上 tools 同样重要:"帮我记一下…" 是聊天式的,不是知识式的,
    所以 `create_note` 在这里也必须可达.
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
    """对 retrieved_docs 重新排序(reranker 在 retriever/tool 内部运行).

    保留为显式 node,一是让 graph 的流程与架构图一致
    (… → rerank_node → generate_node → …),二是便于在生成前裁掉体积过大的
    工具消息.
    """
    hits = state.get("retrieved_docs") or []
    return {"retrieved_docs": hits}


async def generate_node(state: KnowledgeState) -> dict:
    """基于检索到的文本块合成带引用的最终回答."""
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
    """把失败的查询改写(一次),以提高检索召回."""
    original = _latest_user_text(state)
    retry_count = int(state.get("retry_count", 0)) + 1

    rewritten = await _rewrite(original, state)
    # 让 state 的 query 保持为 *改写后* 的查询供检索 node 使用,
    # 同时在 `messages` 里保留原始用户消息作为上下文.
    return {"current_query": rewritten, "retry_count": retry_count}


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _latest_user_text(state: KnowledgeState) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return (msg.content or "").strip()
    return ""


def _active_query(state: KnowledgeState) -> str:
    """驱动本轮检索 *与* 生成的查询.

    一轮开始后 `current_query` 保存的是真实查询,但 `intent_router` 会先往
    那里放一个 `INTENT:<branch>` 标记 —— 因此只有非标记的值才算查询.这正是
    retry 循环得以运转的原因:`rewrite_query_node` 把改写后的查询写进
    `current_query`,检索必须取用它,而不是重读原始用户消息(否则第二次检索
    会与第一次完全相同).

    这里比较的是确切的标记集合,而非前缀测试:被改写的查询本身就可能以
    `INTENT:` 开头(用户完全可能正在问 intent 路由相关的事),把它当作标记
    会静默退回原始查询 —— 重新引入这个 guard 本就是为了防住的 bug.
    """
    current = (state.get("current_query") or "").strip()
    if current and current not in INTENT_MARKERS:
        return current
    return _latest_user_text(state)


def _trim_history(state: KnowledgeState, settings: Settings | None = None) -> list:
    """保留最近 N 轮 human/ai 对话(受预算限制)作为上下文."""
    settings = settings or get_settings()
    budget = settings.max_history_messages
    messages = state.get("messages", [])
    # 排除 ToolMessage(每轮临时的)以及 `knowledge_search` 为携带 tool_calls
    # 而追加的无内容 AIMessage:`ZhipuLLM` 在转成 SDK 载荷时会丢掉
    # tool_calls,转发它只会把一个空的 assistant 轮次摆到模型面前.两者
    # 都不算对话.`isinstance(m.content, str)` 这个 guard 很重要:langchain
    # 允许用 AIMessage(content=[{...}]) 表示工具/多模态轮次,而 list 没有
    # .strip() —— 这类消息必须保留,不能让它把本轮搞崩.
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
    # graph 会在 node 运行前把本轮 HumanMessage 追加到 `messages`,而每个
    # 调用方又会自己重新加上问题(direct_response 里是原文,generate_node
    # 里包装成 "用户问题: …").把它留着会把同一个问题以两种措辞发送两次,
    # 还白占一个 `max_history_messages` 槽位.
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
# 原生 Function Calling(工具循环)
# ---------------------------------------------------------------------------
def _llm_accepts_tools(llm: Any) -> bool:
    """`llm.ainvoke` 是否接受 `tools=` 关键字.

    提前检查,而不是靠捕获 TypeError,这样调用 *内部* 抛出的无关 TypeError
    就不会被误判成 "不支持 tools".
    """
    try:
        return "tools" in inspect.signature(llm.ainvoke).parameters
    except (TypeError, ValueError):
        return False


def _response_tool_calls(response: Any) -> list[dict[str, Any]]:
    """模型请求的 tool calls,已由 LLM 适配器归一化."""
    calls = getattr(response, "tool_calls", None) or []
    return [c for c in calls if isinstance(c, dict)]


def _ai_message_with_calls(response: Any, calls: list[dict[str, Any]]) -> AIMessage:
    """重建请求过工具的 assistant 轮次(用于记录)."""
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
    """执行一个模型请求的 tool call,并把结果作为文本返回.

    按规格,工具失败会以结构化文本回灌给模型,让它自行决定下一步,
    而不是中止整次运行.
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
    """调用模型,执行它请求的任意工具,直到它给出回答.

    当模型不请求工具时 —— 每个不需要工具的查询都是如此 —— 这就是一次与
    无工具实现参数完全相同的调用,因此既有行为不变.

    Args:
        conversation: system + 历史 + 本轮消息.

    Returns:
        `(final_text, transcript)`,其中 transcript 保存要追加到 state 的
        额外 AIMessage/ToolMessage 对(按时间顺序).
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
                # 如果 API 拒绝我们的 tools 载荷,就降级为普通调用,
                # 保证 agent 仍可用,而不是让每个查询都失败.
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
    """把工具的 JSON 字符串结果解析回命中列表."""
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
    """把检索到的文本块渲染成生成提示词的上下文块."""
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
    """生成步骤的 system prompt,已注入上下文."""
    return SYSTEM_PROMPT + f"""

## 本次检索到的知识库内容

以下是针对当前问题的知识库检索结果。请只依据这些内容回答，并按要求标注来源。
如果检索结果与问题无关或为空，请直接说明知识库中未找到相关内容。

--- 检索结果开始 ---
{context}
--- 检索结果结束 ---
"""


async def _classify_intent(query: str) -> str:
    """关闭 thinking 的 GLM 调用:knowledge 还是 direct(无检索)."""
    system = (
        "判断下面这句用户消息是否需要查询个人知识库。"
        "需要检索知识（关于用户个人记录、资料、笔记、文档、过往内容等）返回 knowledge；"
        "仅是寒暄/闲聊/询问助手能力/与知识库无关的开放性问题返回 direct。"
        "只输出 knowledge 或 direct，不要输出其他内容。"
    )
    if DEPS.settings is None or not DEPS.settings.zhipu_api_key:
        raise RuntimeError("未配置 ZHIPU_API_KEY，无法进行分类路由。")
    # 关闭 thinking + 极小的 token 预算,让路由又便宜又快.
    classifier = _make_classifier_llm(DEPS.settings)
    response = await classifier.ainvoke(
        [SystemMessage(content=system), HumanMessage(content=query)]
    )
    label = (response.content or "").strip().lower()
    if label in {BRANCH_KNOWLEDGE, BRANCH_DIRECT}:
        return label
    return BRANCH_KNOWLEDGE


def _make_classifier_llm(settings: Settings) -> Any:
    """构建一个专为廉价,关闭 thinking 的分类调优的 ZhipuLLM."""
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
    """让 GLM 改写一个未能检索到有用内容的查询."""
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
