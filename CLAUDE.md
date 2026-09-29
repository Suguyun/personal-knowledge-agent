# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 仓库结构

git 根目录是 `ai-agent-demo/`，但**所有代码都在 `personal-knowledge-agent/` 下**，所有命令都必须在该目录执行。这里没有可安装的 package，也没有 `-m` 入口：模块之间按顶层名称互相导入（`from config import ...`、`from rag.retriever import Retriever`）。

`Role: Senior AI Agent Architect.md`（仓库根目录）是当初锁定的需求规格（技术栈「已锁定 —— 请勿改动」），实现严格对齐它；`project-architecture.drawio` / `architecture-overview.png` 是对应的架构图。

`docs/` 存放项目文档，文件名格式为 `YYYY-MM-DD-主题.md`（如 `docs/2026-09-24-缺陷修复报告.md`）。

## 常用命令

均在 `personal-knowledge-agent/` 下执行：

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python main.py                          # 交互式对话
python main.py -q "2026年OKR是什么"      # 单次提问
python main.py --rebuild                # 先清空 Chroma 集合重建索引，再进入会话
python main.py --debug                  # 输出 DEBUG 日志

python test_agent.py                    # 事实上的测试入口：写入 2 篇样本文档 + 跑 3 条固定查询
python test_agent.py --offline          # 用 dummy embedder 构建整条流水线，不调用 API/LLM
python test_agent.py --query "..."      # 在 3 条查询之外再跑一条自定义查询
```

没有 pytest/unittest 测试套件，也没有 lint 配置 —— `test_agent.py` 是一个脚本而不是测试运行器，它是事实上的端到端验证手段。只做语法检查可用：
`.venv/bin/python -m py_compile config.py main.py test_agent.py prompts/*.py tools/*.py rag/*.py graph/*.py`

除 `--offline` 外的所有场景都需要 `LLM_API_KEY`；`config.validate_api_key` 会在启动时快速失败。

## 架构

分层结构，依赖自上而下：`config` → `rag` → `tools` → `graph` → `main`。

**图流程**（`graph/builder.py`、`graph/edges.py`）：

```
START → intent_router ─knowledge→ knowledge_search_node → rerank_node → generate_node → END
                      └─direct──→ direct_response_node → END
generate_node ──质量不达标且 retry_count < max_retry──→ rewrite_query_node → knowledge_search_node
```

**`graph/state.py`** 是节点之间的契约：`messages`（通过 `add_messages` reducer 追加）、`retrieved_docs`、`current_query`、`retry_count`、`final_answer`。注意 `current_query` 是一物两用：`intent_router` 把 `INTENT_PREFIX<branch>` 标记塞进该字段，`route_intent` 再解析出来；检索与生成则通过 `nodes._active_query()` 取值 —— 该函数用**精确匹配** `edges.INTENT_MARKERS` 判断"这还是标记吗"，不是前缀测试（否则一个以 `INTENT:` 开头的改写查询会被误判成标记，退回原始查询）。改动路由时请保持这个约定。

**节点依赖注入**（`graph/nodes.py`）：节点只接收 `state`，协作者从模块级的 `DEPS` 容器读取，由 `build_graph` 调用 `nodes._set_deps(...)` 填入。重构前需要知道的后果：这套装配本质是全局可变状态，所以一个进程内实际只能有一个 graph，两个使用不同 retriever/LLM 的已编译 graph 无法共存；`build_graph` 必须先于任何节点执行。

**LLM 调用**（`graph/llm.py`）：所有调用都走 `OpenAICompatLLM` 适配器，底层是 `openai` SDK 的 `AsyncOpenAI`（原生异步，不再需要 `asyncio.to_thread`），不走 `langchain_openai`。适配器提供 `ainvoke(messages, tools) -> 带 .content / .tool_calls 的对象`，把 langchain 消息转成普通 dict。节点用 `_as_text` 统一取文本。**服务商完全由配置决定**（`llm_base_url` / `llm_model` / `llm_api_key`）：DeepSeek 与智谱 BigModel 都提供 OpenAI 兼容端点，换厂商只改 `.env`，不动节点代码。`config.py` 的代码默认值仍是智谱 GLM-5.2（对齐架构文档的锁定选型），实际跑的由 `.env` 覆盖。

带思考的模型（GLM-5.2 默认开启；DeepSeek 的 `thinking` 模式亦然）把推理轨迹放在 `reasoning_content`，因此 `max_tokens` 默认 8192 —— token 预算给小了，思考过程会吃掉全部额度，最终 `content` 会是**空字符串**。`thinking` 是非标准参数，只能经 `extra_body` 下发；做廉价路由时，`nodes._make_classifier_llm` 会另建一个 `thinking={"type": "disabled"}`、`max_tokens=16` 的适配器（不关思考的话那 16 个 token 会被推理轨迹吃光，分类恒为空）。

**`reasoning_content` 的回传契约（换服务商时最容易踩的坑）**：请求带 `tools=` 时，历史轮次的 `reasoning_content` 必须原样回传，否则 API 直接拒绝该请求；不带 `tools=` 时该字段会被忽略。因此 `nodes._ai_message_with_calls` 会把响应的 `reasoning_content` 存进 `AIMessage.additional_kwargs`（checkpointer 会持久化它），`llm._to_dict` 再取回来放进请求。`nodes._trim_history` 会丢弃**任何**带 `tool_calls` 的 assistant 轮次 —— 它的 `ToolMessage` 在同一处也被剥掉了，留着就会变成"有 tool_calls 却没有工具结果"的非法载荷。

按规格要求，提示词里**不得**出现 CoT/ReAct/"think step by step" 之类的推理指令（`prompts/system_prompt.py`）—— 所用模型自带思考能力（GLM-5.2 默认开启，DeepSeek 的 thinking 模式亦然）。

**检索**（`rag/`）：`Retriever.search` 先嵌入查询，从 Chroma 取 `top_k=10`，重排后返回 `rerank_top_k=5`；重排器关闭或加载失败时降级为原始相似度分数。重排发生在 `Retriever` **内部**，所以 `rerank_node` 是有意为之的透传节点，只为对齐规格里的流程图 —— 不要"顺手修好"把逻辑挪进去。`Retriever` 是通过 `store.embedder` 拿到 embedder 的，因此构造时应传入共享的 `VectorStore`。

**工具**（`tools/`）是闭包工厂（`_make_*`），由 `get_tools(settings, retriever, store)` 在运行时绑定依赖，再经 `@tool` 装饰。它们的 docstring 特意写成中文 —— 那就是给模型看的工具 schema。工具返回的是 JSON **字符串**，由 `nodes._parse_hits` 解析回来（`knowledge_search` 节点直接调用的那条路径）。`create_note` 会写入 `data/notes/<title>.md` 并通过 `VectorStore.reindex_path()` 立即建索引（按 `path` 替换自身旧片段，同标题重存不会产生重复或过期片段）。

**原生 Function Calling** 由 `nodes._tool_enabled_completion()` 实现：它把 `DEPS.tools` 交给 `OpenAICompatLLM`（内部序列化成 OpenAI `tools=` 载荷），若模型返回 `tool_calls` 就逐个执行（`_execute_tool_call`，失败以文本回灌给模型，见规格"structured error → LLM decides"），把 `AIMessage`/`ToolMessage` 追加进对话后再次调用，直到模型给出纯文本答案；轮次上限为 `settings.max_tool_iterations`（默认 3，耗尽时返回明确提示而非静默收尾）。`generate_node` 与 `direct_response` 都走这条路径 —— 后者同样必要，因为"帮我记一下"会被判成 chat 分支。**模型不请求工具时只发一次 API 调用，参数与旧的纯实现完全一致**；若 API 拒绝我们的 tools 载荷，会自动降级为不带工具的一次调用（打 warning）。`_llm_accepts_tools()` 用签名检查判断 LLM 是否支持 `tools=`，因此传入自定义/桩 LLM 也能工作。

## 配置与数据

`config.Settings`（pydantic-settings + `config.py` 同级的 `.env`）是唯一配置源，被 `rag`、`tools`、`graph` 共同读取。从 `.env.example` 复制出 `.env`。

**必须在 `personal-knowledge-agent/` 下运行。** `.env` 里把路径写成 `KB_DIR=./data/kb`，pydantic 会加载成**相对路径**，也就是 KB/notes/chroma 的实际位置跟随进程 cwd，而不是代码目录。

`EMBEDDING_BACKEND` 可选 `local`（sentence-transformers 的 `BAAI/bge-small-zh-v1.5`，完全离线，本地当前用的值）或 `zhipu`（通过 OpenAI 兼容接口调 `embedding-3`，需要单独购买 embedding 资源包）。代码默认值是 `zhipu`；`.env`/`.env.example` 设为 `local`。**切换后端会改变向量维度，已有 embedding 会失效 —— 换后端后一定要跟一次 `--rebuild`。** 另外 `config.py` 在 `HF_ENDPOINT` 未设置时默认指向 `hf-mirror.com`。本地 `RERANKER_ENABLED=false`（该模型需下载约 2GB）。

索引是"全量重建"模型：启动时仅在集合为空时建索引，`data/kb/` 中被删除或修改的文件永远不会被清理，`--rebuild`（内部调用 `store.reset()`）是唯一刷新方式。笔记是例外：`create_note` 走 `VectorStore.reindex_path()`，按 `path` 替换自身旧片段。`data/kb/*.md` 被 git 跟踪；`data/` 下其余内容（chroma、`checkpoints.sqlite`、notes）被忽略。

## 异步与 checkpoint 约束

所有节点都是 async，这决定了 checkpointer 的选型：`SqliteSaver` 在 `ainvoke` 下会抛 `NotImplementedError`，所以用的是 `AsyncSqliteSaver`，并且每个节点都必须保持 async —— 如果你把节点改成同步，就要换回 `SqliteSaver`。

`build_graph` 自己打开 aiosqlite 连接，而没有用 `AsyncSqliteSaver.from_conn_string`（那是个 async 上下文管理器，会在 `async with` 结束时关闭连接，从而破坏整个 graph 生命周期的 checkpoint）。连接被挂到 `compiled._checkpointer_conn`；短生命周期的 `asyncio.run` 脚本必须在事件循环关闭前 `await close_graph(graph)`，否则 aiosqlite 的工作线程会报 "Event loop is closed"。`build_graph_sync` 只适用于普通脚本 —— 它返回的 graph 绑定在构造它的那个事件循环上。

多轮对话的连续性来自按 `thread_id` 索引的 checkpointer，而不是进程内保存的任何状态。REPL 用固定 `cli-session`（跨重启保留，属有意设计）；`-q` 与 `test_agent.py` 每次运行使用一次性 `one-shot-<uuid8>` / `test-<uuid8>`，并在结束时调用 `clear_thread(graph, thread_id)` 删掉该线程 —— 唯一 id 保证历史隔离，删除则避免 checkpoint 库随每次调用无限增长（该库没有任何自动清理）。

## 已知缺陷

**工具调用循环已经过真实 API 验证（2026-09-29，DeepSeek `deepseek-flash`）。** 实测确认：模型会按要求发起工具调用（日志 `Tool iteration 1/3`）、参数能通过 `args_schema` 校验、`create_note` 真的落盘写入了文件、带 `tools=` 的后续请求被 API 接受（即 `reasoning_content` 回传契约成立）、全程无 `retrying without tools` 降级。序列化（`_tools_to_payload`、`_to_dict` 的 `tool_calls` 往返、畸形 `arguments` 判定）与循环控制（不请求工具时单次调用、未知工具、轮次上限、无 `tools=` 参数的 LLM 降级）另有桩件覆盖。

但这是**一次性的实测观察，不是回归测试** —— 仓库里没有测试套件，改动这条链路（尤其是换服务商）后仍需用真实 API 重新验证一次，重点是带 `tools=` 的请求能否正确回传 `reasoning_content`。

**检索仍是"全量重建"。** 启动时仅在集合为空时建索引，`data/kb/` 中被修改或删除的文件不会被刷新；`--rebuild` 是唯一刷新手段，而它会连笔记一起清空。`VectorStore.reindex_path()` 已能按 `path` 安全替换单个文件的片段，但目前只有 `create_note` 用它。

**`path` 元数据是相对路径。** 存量片段的 `path` 形如 `data/kb/xxx.md`，取决于建索引时的 cwd；换个目录运行会让按 `path` 的替换匹配不到旧片段。见「配置与数据」一节的 cwd 规则。

## 依赖版本陷阱

- `openai` 同时服务 LLM 与 zhipu embedding 后端，不需要智谱官方的 `zai-sdk`（已移除）。
- **`numpy<2` 是硬约束**：torch 2.2.x 与 numpy 2.x 的 ABI 不兼容，sentence-transformers 会报 "Numpy is not available"。
- `transformers` 锁在 `<5`，因为 5.x 要求 torch ≥ 2.4。
- 本地 venv 为 Python 3.11.4。
