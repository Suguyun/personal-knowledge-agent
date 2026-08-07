"""Edge / routing logic for the graph.

Three routing decisions live here:

1. `route_intent`        — which branch the intent_router picked
2. `route_after_generate`— whether to accept the answer or retry with a
                           rewritten query (quality check + retry budget)
3. `route_after_rewrite` — after rewriting, whether to re-run retrieval
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from config import Settings, get_settings
from graph.state import KnowledgeState

# Literal branch names, shared with builder.py.
BRANCH_KNOWLEDGE = "knowledge"
BRANCH_DIRECT = "direct"


def route_intent(state: KnowledgeState) -> str:
    """Decide which branch the user query takes.

    The intent_router node stores its decision on `current_query` in the form
    `INTENT:<knowledge|direct>`; this function strips the prefix and maps it
    onto the graph branch names. Any unknown value defaults to the knowledge
    branch (retrieval is the safe default).
    """
    marker = state.get("current_query", "")
    if marker.startswith("INTENT:"):
        intent = marker.split(":", 1)[1].strip().lower()
        if intent == BRANCH_DIRECT:
            return BRANCH_DIRECT
    return BRANCH_KNOWLEDGE


def route_after_generate(state: KnowledgeState, settings: Settings | None = None) -> str:
    """Quality gate: accept the answer, or rewrite the query and retry once.

    Conditions that trigger a retry:
        - retrieval returned nothing for this turn, OR
        - the model produced an explicit "not found" answer

    Only retried when `retry_count < settings.max_retry` (default 1).

    Returns:
        "rewrite" to loop back through retrieval, else "end".
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
    """After rewriting the query, always re-enter retrieval.

    Kept as a named function so the graph wiring stays declarative and the
    retry budget bookkeeping (incremented inside rewrite_query_node) is easy
    to inspect.
    """
    return "knowledge_search"


__all__ = [
    "BRANCH_KNOWLEDGE",
    "BRANCH_DIRECT",
    "route_intent",
    "route_after_generate",
    "route_after_rewrite",
]
