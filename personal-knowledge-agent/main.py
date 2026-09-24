"""Interactive CLI for the personal knowledge agent.

Usage:
    python main.py                 # interactive REPL (default)
    python main.py --query "..."   # single-shot query
    python main.py --rebuild       # wipe + rebuild the index before the session

On startup the CLI indexes every markdown/text file in `KB_DIR` (idempotent —
files already indexed are skipped via the persistent collection count check),
then drops into a REPL. `Ctrl+C` or `/exit` leaves the session.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid

from config import Settings, get_settings, validate_api_key
from graph.builder import build_graph, clear_thread, close_graph
from rag.loader import load_documents
from rag.splitter import split_documents
from rag.vectorstore import VectorStore

logger = logging.getLogger(__name__)

_SPECIAL_COMMANDS = {"/exit", "/quit", "/bye", "exit", "quit"}


def _setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Keep the noisy HTTP/urllib loggers quiet by default.
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def index_knowledge_base(settings: Settings, store: VectorStore) -> int:
    """Load KB files, split, and index any new chunks (idempotent).

    Returns the number of chunks added during this run.
    """
    docs = load_documents(settings.kb_dir)
    if not docs:
        logger.info("知识库目录 %s 中没有可索引的文档。", settings.kb_dir)
        return 0

    chunks = split_documents(
        docs,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if store.count() > 0:
        # Idempotency heuristic: we don't track per-file hashes; a full rebuild
        # (--rebuild) is the explicit way to refresh. Skip re-adding when the
        # collection is already populated so restarts are fast.
        logger.info(
            "集合已有 %d 个片段，跳过增量索引（如需重建请加 --rebuild）。",
            store.count(),
        )
        return 0

    added = store.add_documents(chunks)
    logger.info("索引完成: 新增 %d 个片段。", added)
    return added


async def _run_session(settings: Settings, graph) -> None:
    """Interactive REPL."""
    print()
    print("=" * 60)
    print(" 个人知识助手已就绪 (GLM-5.2 + LangGraph + ChromaDB)")
    print(" 输入问题开始对话；输入 /exit 退出。")
    print("=" * 60)
    thread_id = "cli-session"

    while True:
        try:
            query = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not query:
            continue
        if query.lower() in _SPECIAL_COMMANDS:
            print("再见！")
            break

        print("\n助手: ", end="", flush=True)
        try:
            from langchain_core.messages import HumanMessage

            state = {"messages": [HumanMessage(content=query)], "retry_count": 0}
            result = await graph.ainvoke(
                state,
                config={"configurable": {"thread_id": thread_id}},
            )
            print(result.get("final_answer") or "(无回答)")
        except Exception as exc:
            logger.exception("Agent run failed")
            print(f"（运行出错: {exc}）")
            continue


async def _run_query(settings: Settings, graph, query: str) -> None:
    from langchain_core.messages import HumanMessage

    # A one-shot run must not inherit history from earlier runs: the
    # checkpointer persists to SQLite, so a fixed thread_id would load the
    # previous invocation's conversation and feed it back in as context.
    thread_id = f"one-shot-{uuid.uuid4().hex[:8]}"

    print("问题:", query)
    print("助手: ", end="", flush=True)
    state = {"messages": [HumanMessage(content=query)], "retry_count": 0}
    try:
        result = await graph.ainvoke(
            state,
            config={"configurable": {"thread_id": thread_id}},
        )
        print(result.get("final_answer") or "(无回答)")
    finally:
        # The id is unique per invocation, so nothing would ever reuse or
        # prune it — drop the thread rather than leaking it into the DB.
        await clear_thread(graph, thread_id)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Personal knowledge agent")
    parser.add_argument("--query", "-q", help="单次提问，运行后退出")
    parser.add_argument("--rebuild", action="store_true", help="先清空向量库并重建索引")
    parser.add_argument("--debug", action="store_true", help="输出调试日志")
    args = parser.parse_args()

    _setup_logging(debug=args.debug)
    settings = get_settings()
    validate_api_key(settings)
    settings.ensure_dirs()

    store = VectorStore(settings)
    if args.rebuild:
        logger.info("重建索引: 清空现有集合…")
        store.reset()
        index_knowledge_base(settings, store)
    else:
        index_knowledge_base(settings, store)

    graph = await build_graph(settings=settings, store=store)

    try:
        if args.query:
            await _run_query(settings, graph, args.query)
        else:
            await _run_session(settings, graph)
    finally:
        # Close the aiosqlite checkpoint connection before the event loop
        # shuts down, otherwise its worker thread raises "Event loop is
        # closed" on exit (see README: close_graph).
        await close_graph(graph)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as exc:
        print(f"启动失败: {exc}", file=sys.stderr)
        sys.exit(1)
