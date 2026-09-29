"""个人知识助手的交互式 CLI.

用法:
    python main.py                 # 交互式 REPL(默认)
    python main.py --query "..."   # 单次提问
    python main.py --rebuild       # 会话开始前清空并重建索引

启动时 CLI 会索引 `KB_DIR` 下所有 markdown/text 文件(幂等 — 已索引的文件
通过持久化的集合数量检查跳过),然后进入 REPL.`Ctrl+C` 或 `/exit` 退出会话.
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
    # 默认压住吵闹的 HTTP/urllib logger.
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def index_knowledge_base(settings: Settings, store: VectorStore) -> int:
    """加载 KB 文件,切分并索引新增的 chunk(幂等).

    返回本次运行新增的 chunk 数量.
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
        # 幂等启发式:我们不跟踪逐文件 hash;全量重建(--rebuild)才是显式的
        # 刷新方式.集合已有数据时跳过重复添加,让重启更快.
        logger.info(
            "集合已有 %d 个片段，跳过增量索引（如需重建请加 --rebuild）。",
            store.count(),
        )
        return 0

    added = store.add_documents(chunks)
    logger.info("索引完成: 新增 %d 个片段。", added)
    return added


async def _run_session(settings: Settings, graph) -> None:
    """交互式 REPL."""
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

    # 单次运行不能继承此前运行的历史:checkpointer 持久化到 SQLite,固定
    # thread_id 会加载上一次调用的对话,并把它作为上下文回灌.
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
        # 该 id 每次调用都唯一,不会被复用或清理 — 主动删掉这个 thread,
        # 而不是把它泄漏进 DB.
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
        # 在事件循环关闭前关掉 aiosqlite checkpoint 连接,否则退出时它的
        # worker thread 会抛 "Event loop is closed"(见 README:close_graph).
        await close_graph(graph)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as exc:
        print(f"启动失败: {exc}", file=sys.stderr)
        sys.exit(1)
