"""Graph 的 edge / 路由逻辑.

这里有三个路由决策:

1. `route_intent`        — intent_router 选了哪条分支
2. `route_after_generate`— 是接受回答,还是用改写后的查询 retry
                           (质量检查 + retry 预算)
3. `route_after_rewrite` — 改写之后,是否重新执行检索
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from config import Settings, get_settings
from graph.state import KnowledgeState

# 字面量分支名,与 builder.py 共享.
BRANCH_KNOWLEDGE = "knowledge"
BRANCH_DIRECT = "direct"

# `intent_router` 把它的决策暂存在 `current_query` 上,形式为
# f"{INTENT_PREFIX}<branch>".前缀在这里定义,写入方(graph/nodes.py)
# 与读取方(本模块)都用它,因此标记格式确实只在一处声明 ——
# 改动它不会让两边失配.
INTENT_PREFIX = "INTENT:"

# `intent_router` 可能写入的确切标记值.匹配这些完整字符串
# (而不是测试前缀)可以避免一个恰好以该前缀开头的改写查询
# 被误判成标记.
INTENT_MARKERS = frozenset(
    f"{INTENT_PREFIX}{branch}" for branch in (BRANCH_KNOWLEDGE, BRANCH_DIRECT)
)


def route_intent(state: KnowledgeState) -> str:
    """决定用户查询走哪条分支.

    intent_router node 把它的决策以 `INTENT:<knowledge|direct>` 的形式存到
    `current_query` 上;本函数剥掉前缀并将其映射到 graph 的分支名.任何未知
    值都默认走 knowledge 分支(检索是安全的默认选项).
    """
    marker = state.get("current_query", "")
    if marker.startswith(INTENT_PREFIX):
        # 按前缀长度切片,而不是用 ":" 分割:分隔符是 INTENT_PREFIX
        # 结尾的那个字符,而非硬编码的冒号.
        intent = marker[len(INTENT_PREFIX):].strip().lower()
        if intent == BRANCH_DIRECT:
            return BRANCH_DIRECT
    return BRANCH_KNOWLEDGE


def route_after_generate(state: KnowledgeState, settings: Settings | None = None) -> str:
    """质量门禁:接受回答,或者改写查询并 retry 一次.

    触发 retry 的条件:
        - 本轮检索没有返回任何内容,或者
        - 模型给出了明确的 "not found" 回答

    仅当 `retry_count < settings.max_retry`(默认 1)时才 retry.

    Returns:
        "rewrite" 表示回到检索再走一遍,否则 "end".
    """
    settings = settings or get_settings()

    retrieved = state.get("retrieved_docs") or []
    answer = (state.get("final_answer") or "").strip()
    retry_count = int(state.get("retry_count", 0))

    retrieval_empty = len(retrieved) == 0
    looks_not_found = any(token in answer for token in ("未找到", "没有找到", "not found", "未收录"))

    if (retrieval_empty or looks_not_found) and retry_count < settings.max_retry:
        return "rewrite"

    return "end"


def route_after_rewrite(state: KnowledgeState) -> str:
    """改写查询后,总是重新进入检索.

    保留为具名函数,是为了让 graph 的接线保持声明式,并让 retry 预算的记账
    (在 rewrite_query_node 内自增)便于检查.
    """
    return "knowledge_search"


__all__ = [
    "BRANCH_KNOWLEDGE",
    "BRANCH_DIRECT",
    "INTENT_PREFIX",
    "INTENT_MARKERS",
    "route_intent",
    "route_after_generate",
    "route_after_rewrite",
]
