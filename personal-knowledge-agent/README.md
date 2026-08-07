# 个人知识助手 (Personal Knowledge Agent)

一个基于 **GLM-5.2 + LangGraph + ChromaDB** 的个人知识库问答 Agent。
它检索你的本地 markdown 笔记/文档，组织成带引用的回答，并支持把新内容沉淀回知识库。

## 技术栈

| 组件 | 选型 |
|---|---|
| LLM | GLM-5.2（智谱官方 `zai-sdk`, `ZhipuAiClient`） |
| 编排 | LangGraph（StateGraph, 异步节点, SQLite Checkpointer） |
| 向量库 | ChromaDB（本地持久化） |
| Embedding | 默认本地 `BAAI/bge-small-zh-v1.5`（离线, 不耗 API 额度） |
| Rerank | `bge-reranker-v2-m3`（CrossEncoder, 可降级） |
| 工具协议 | 原生 Function Calling（`@tool` + `bind_tools`） |
| 配置 | `python-dotenv` + `pydantic-settings` |

### LLM 适配（graph/llm.py）

LLM 调用走智谱官方 `zai-sdk` 的 `ZhipuAiClient`，封装为 `ZhipuLLM` 适配器
（`ainvoke(messages) -> .content`）。GLM-5.2 默认开启思考（thinking），
适配器使用充足 `max_tokens`（默认 8192），避免思考过程耗尽 token 预算导致
回答为空。意图分类节点使用 thinking 关闭 + 小 `max_tokens` 的轻量调用。

### Embedding 后端

`EMBEDDING_BACKEND` 二选一：

- `local`（默认）：sentence-transformers 加载 `LOCAL_EMBEDDING_MODEL`
  （默认 `BAAI/bge-small-zh-v1.5`），完全离线、不消耗智谱 embedding 资源包。
- `zhipu`：智谱 `embedding-3`（批量 `embedding_batch_size`=20），需要账户有
  embedding 资源包（与聊天资源包分开计费）。

## 快速开始

### 1. 安装依赖

```bash
cd personal-knowledge-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 ZHIPU_API_KEY（https://open.bigmodel.cn 获取）
```

### 3. 放入知识库文档

把个人 markdown / txt 文档放进 `data/kb/`（默认路径，可用 `KB_DIR` 覆盖）。
目录不存在会自动创建；启动时自动加载并索引。

### 4. 运行

```bash
python main.py                # 交互式对话
python main.py -q "2026年OKR是什么"   # 单次提问
python main.py --rebuild      # 先重建向量索引再启动
```

运行测试脚本（自动写入两个示例文档并跑 3 条查询）：

```bash
python test_agent.py
python test_agent.py --offline   # 不调用 LLM，只验证流水线可构建
```

## 图流程

```
START → intent_router ──knowledge──→ knowledge_search_node
                    └──direct──→ direct_response_node → END
knowledge_search_node → rerank_node → generate_node
generate_node ──质量达标──→ END
generate_node ──质量不达标且 retry<1──→ rewrite_query_node → knowledge_search_node（再检索一次）
```

- `intent_router`：`thinking` 关闭的轻量 GLM 调用，把问题分流为"需要检索"与"直接回答"。
- 检索 0 结果或模型给出"未找到"且重试次数未满时，自动改写查询再检索一次。
- 多轮对话通过 SQLite Checkpointer 按 `thread_id` 持久化。

## 工具

| 工具 | 作用 |
|---|---|
| `knowledge_search(query, filters?)` | 检索知识库，返回 top-5 片段（含来源） |
| `list_documents()` | 列出全部文档及章节/更新时间概览 |
| `create_note(title, content)` | 写入 `data/notes/` 并立即加入索引 |

## 引用与规则

- 回答严格基于检索结果，附带 `[来源: 文档名, 章节]` 标注。
- 检索不到即如实说明"知识库中未找到"，绝不编造（零幻觉）。
- 系统提示词不含任何 CoT / ReAct 指令 —— GLM-5.2 默认自带思考能力。

## 依赖版本注意

- **`zai-sdk`**：智谱官方 SDK（`from zai import ZhipuAiClient`）。注意 PyPI 上还有一个
  **空包 `zai`**（0.0.2，无任何代码），不要装错 —— 正确包名是 `zai-sdk`。
- **numpy 必须 `<2`**：本 venv 的 torch 为 2.2.x，与 numpy 2.x **ABI 不兼容**
  （sentence-transformers 会报 `Numpy is not available`）。请保持 `numpy<2`。
- **HuggingFace 国内镜像**：`config.py` 默认设置 `HF_ENDPOINT=https://hf-mirror.com`，
  本地 embedding / reranker 模型可正常下载；如自行指定 HF_ENDPOINT 会覆盖。
- `langgraph-checkpoint-sqlite` 的 `SqliteSaver` **不支持 async 方法**。本实现采用
  `AsyncSqliteSaver`（需 `aiosqlite`），因为所有节点都是 async。若你改为 sync 节点，
  需同步改回 `SqliteSaver`。
- reranker 依赖 `sentence-transformers` → `transformers`。**transformers 5.x 要求
  torch ≥ 2.4**；如果安装时解析到 torch 2.2.x，请将 transformers 降到 `<5`（已在
  requirements.txt 中约束）。若没有 GPU / 不想下载模型，设 `RERANKER_ENABLED=false`，
  流水线会自动降级为原始向量分数，不影响使用。

## 已知限制

1. **索引为"全量重建"模型**：启动时仅当集合为空才索引；增删文档后用 `--rebuild` 刷新。
   （未做文件指纹去重，重复运行不会重复插入，但删除文档不会自动清理。）
2. **单用户本地场景**：无鉴权、无多租户；向量库与笔记均为本地明文。
3. **Reranker 需要下载模型**（bge-reranker-v2-m3 约 2GB，首次调用时下载）。
   若失败/禁用会自动降级为原始向量分数，仅日志警告。
4. **重试仅一次**：查询改写后仍无结果则直接给出"未找到"答复。
5. **GLM 错误码处理**：余额不足(1301)/限流(1305) 在 `config` 中登记，
   当前实现里 API 异常统一由 OpenAI SDK 的重试与节点级异常兜底；如需更细粒度
   可按错误码在 `nodes.py` 增加专门重试逻辑。
6. **`build_graph` 是 async 的**：`AsyncSqliteSaver` 必须在事件循环内构造。
   在 async 上下文用 `await build_graph(...)`；脚本场景用 `build_graph_sync()`。

## 下一步建议

- 文件指纹（hash）增量索引 + `delete` API 清理已删除文档。
- 引入 `gradio`/FastAPI Web 界面，或接 `wecom`/飞书机器人。
- 按主题给笔记打标签，支持 `filters={"tags": ...}` 的元数据过滤。
- 检索命中后缓存 embeddings（`embedding-3` 计费）。
- 接入 `xinference` 自托管 reranker 以彻底摆脱下载依赖。
