"""End-to-end test for the personal knowledge agent.

Usage:
    python test_agent.py               # full run: seed KB + index + 3 queries
    python test_agent.py --offline     # build the pipeline, skip live LLM calls
    python test_agent.py --query "..." # run a single extra query

The test seeds `data/kb/` with two sample markdown documents (skipped if they
already exist so your real notes are never touched), indexes the KB, then runs
three fixed queries and prints each answer.

Requires ZHIPU_API_KEY unless `--offline` is given.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from config import Settings, get_settings
from graph.builder import build_graph
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
    """Write the two sample documents into data/kb/ (no-op if present)."""
    settings.kb_dir.mkdir(parents=True, exist_ok=True)
    for name, content in _SAMPLE_DOCS.items():
        path = settings.kb_dir / name
        if not path.exists():
            path.write_text(content.lstrip("\n"), encoding="utf-8")
            print(f"[seed] 已写入示例文档: {path.name}")
        else:
            print(f"[seed] 已存在，跳过: {path.name}")


class _DummyEmbedder:
    """Deterministic 8-dim vectors so the pipeline runs fully offline.

    Used only by `--offline`, where no API key is required and no network
    calls happen. Vectors are content-derived so retrieval ordering is
    deterministic and reproducible.
    """

    def __init__(self) -> None:
        self.dim = 8

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            # Sum of codepoints as a simple hashed feature vector.
            vec = [0.0] * self.dim
            for i, ch in enumerate(text):
                vec[i % self.dim] += ord(ch)
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([round(v / norm, 6) for v in vec])
        return out


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

    graph = await build_graph(settings=settings)
    try:
        for q in queries:
            print("\n" + "-" * 64)
            print(f"查询: {q}")
            from langchain_core.messages import HumanMessage

            result = await graph.ainvoke(
                {"messages": [HumanMessage(content=q)], "retry_count": 0},
                config={"configurable": {"thread_id": "test"}},
            )
            answer = result.get("final_answer") or "(无回答)"
            print(f"回答:\n{answer}")
    finally:
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
        _index(settings, offline=True)
        from graph.builder import build_graph_sync

        build_graph_sync(settings=settings)
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
