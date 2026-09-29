"""个人知识助手的端到端测试.

用法:
    python test_agent.py               # 完整运行:写入样例 KB + 建索引 + 3 条查询
    python test_agent.py --offline     # 只构建流水线,跳过真实 LLM 调用
    python test_agent.py --query "..." # 额外执行一条查询

测试会向 `data/kb/` 写入两篇样例 markdown 文档(若已存在则跳过,绝不触碰你的
真实笔记),建立 KB 索引,然后跑三条固定查询并打印每个回答.

除非给出 `--offline`,否则需要 ZHIPU_API_KEY.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

from config import Settings, get_settings
from graph.builder import build_graph, clear_thread
from rag.loader import load_documents
from rag.splitter import split_documents
from rag.vectorstore import VectorStore

_SAMPLE_DOCS = {
    "AI编码工具.md": """\
# AI 编码工具

## 2026年工具选型

个人日常使用的主要 AI 编码工具有三个：Claude Code、Cursor 和 Continue。
Claude Code 是命令行界面，擅长多文件重构与自动化任务。
Cursor 是图形界面编辑器，补全速度最快。
Continue 是开源插件，可自由更换底层模型。

## 使用技巧

长任务优先用 Claude Code，因为它有较好的规划和 agent 能力。
短任务或日常改 bug 用 Cursor 更顺手。
使用 Continue 时建议将模型切换为 GLM-5.2，成本最低。
""",
    "会议记录-2026Q2.md": """\
# 2026 Q2 团队会议记录

## 7月1日 周会

本次会议确认了季度 OKR：知识管理平台上线、内部工具链统一。
平台选型定为：LangGraph 做流程编排、ChromaDB 存向量、GLM-5.2 提供推理。
负责人：小张负责 RAG 检索优化，小李负责前端。

## 7月8日 评审会

评审结论：知识库检索准确率需要达到 85% 以上才能上线。
主要瓶颈是重排序环节，决定引入 bge-reranker-v2-m3 提升精度。
会上还决定每周四下午更新知识库文档。
""",
}


def seed_kb(settings: Settings) -> None:
    """把两篇样例文档写入 data/kb/(已存在则不做任何事)."""
    settings.kb_dir.mkdir(parents=True, exist_ok=True)
    for name, content in _SAMPLE_DOCS.items():
        path = settings.kb_dir / name
        if not path.exists():
            path.write_text(content.lstrip("\n"), encoding="utf-8")
            print(f"[seed] 已写入示例文档: {path.name}")
        else:
            print(f"[seed] 已存在，跳过: {path.name}")


class _DummyEmbedder:
    """确定性的 8 维向量,让流水线可以完全离线运行.

    仅由 `--offline` 使用,此时不需要 API key,也不会发生任何网络调用.向量由
    内容派生,因此检索顺序是确定且可复现的.
    """

    def __init__(self) -> None:
        self.dim = 8

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            # 用 codepoint 之和作为简单的哈希特征向量.
            vec = [0.0] * self.dim
            for i, ch in enumerate(text):
                vec[i % self.dim] += ord(ch)
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([round(v / norm, 6) for v in vec])
        return out


def _offline_settings(settings: Settings) -> tuple[Settings, Path]:
    """指向一次性存储的 settings,以及需要删除的临时根目录.

    离线运行会索引确定性的 8 维 dummy 向量.把它们写进真实集合会让集合不可用:
    下一次真实运行看到集合非空,跳过建索引,随后每条查询都会因维度不匹配而失败
    (512 维本地 embedding 对 8 维已存向量),直到有人跑 `--rebuild`.
    checkpoint DB 出于同样的原因被重定向 — 离线路径不该碰生产库
    (`data/checkpoints.sqlite`).
    """
    tmp_root = Path(tempfile.mkdtemp(prefix="pka-offline-"))
    offline = settings.model_copy(
        update={
            "chroma_db_dir": tmp_root / "chroma",
            "chroma_collection_name": "personal_knowledge_offline",
            "sqlite_checkpoint_path": tmp_root / "checkpoints.sqlite",
        }
    )
    return offline, tmp_root


async def _build_and_close(settings: Settings) -> None:
    """构建流水线并干净地拆除(离线冒烟测试).

    使用 `await build_graph(...)` + `close_graph` 而不是 `build_graph_sync`,
    因为后者包装了 `asyncio.run`,会在 aiosqlite worker thread 仍存活时关闭
    事件循环 — 这正是 sync helper 文档中标注为不安全的拆除方式.
    """
    from graph.builder import close_graph

    graph = await build_graph(settings=settings)
    await close_graph(graph)


def _index(settings: Settings, offline: bool = False) -> int:
    store = VectorStore(settings, embedder=_DummyEmbedder() if offline else None)
    docs = load_documents(settings.kb_dir)
    chunks = split_documents(
        docs, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    if store.count() > 0:
        print(f"[index] 集合已有 {store.count()} 个片段，跳过重建。")
        return 0
    n = store.add_documents(chunks)
    print(f"[index] 新增 {n} 个片段。")
    return n


async def _run(settings: Settings, queries: list[str]) -> None:
    from graph.builder import close_graph

    # 每次运行用全新的 thread,让三条查询彼此共享上下文,但绝不与之前的
    # `test_agent.py` 运行共享(checkpointer 是持久化的,固定 thread_id
    # 会重放旧回答).
    thread_id = f"test-{uuid.uuid4().hex[:8]}"

    graph = await build_graph(settings=settings)
    try:
        for q in queries:
            print("\n" + "-" * 64)
            print(f"查询: {q}")
            from langchain_core.messages import HumanMessage

            result = await graph.ainvoke(
                {"messages": [HumanMessage(content=q)], "retry_count": 0},
                config={"configurable": {"thread_id": thread_id}},
            )
            answer = result.get("final_answer") or "(无回答)"
            print(f"回答:\n{answer}")
    finally:
        # 不要把本次运行独有的 thread 留在 checkpoint DB 中.
        await clear_thread(graph, thread_id)
        await close_graph(graph)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the personal knowledge agent")
    parser.add_argument("--offline", action="store_true",
                        help="只构建流水线，不调用 LLM")
    parser.add_argument("--query", help="额外执行一条自定义查询")
    args = parser.parse_args()

    settings = get_settings()
    seed_kb(settings)

    if args.offline:
        offline_settings, tmp_root = _offline_settings(settings)
        print(f"[offline] 使用临时向量库: {offline_settings.chroma_db_dir}")
        try:
            _index(offline_settings, offline=True)
            asyncio.run(_build_and_close(offline_settings))
        finally:
            # 冒烟测试会被反复运行;没有这一步,每次运行都会留下一个
            # pka-offline-* 目录.
            shutil.rmtree(tmp_root, ignore_errors=True)
        print("\n[offline] 流水线构建成功（未调用任何 API/LLM）。")
        return

    _index(settings)

    from config import validate_api_key

    try:
        validate_api_key(settings)
    except RuntimeError as exc:
        print(f"\n缺少 API Key: {exc}\n请先设置 ZHIPU_API_KEY，或使用 --offline。",
              file=sys.stderr)
        sys.exit(1)

    queries = [
        "我日常使用哪些 AI 编码工具？各有什么特点？",
        "2026 Q2 知识管理平台的技术选型是什么？",
        "知识库检索准确率的目标是多少？",
    ]
    if args.query:
        queries.append(args.query)

    asyncio.run(_run(settings, queries))


if __name__ == "__main__":
    main()
