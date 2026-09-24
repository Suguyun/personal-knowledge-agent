# 任务：实现个人知识助手 Agent

## 技术栈（已锁定 —— 请勿改动）
- LLM：GLM-5.2，走 OpenAI 兼容 API（base_url: https://open.bigmodel.cn/api/paas/v4）
- 框架：LangGraph（Python，StateGraph）
- 向量库：ChromaDB（本地持久化）
- Embedding：zhipuai embedding-3（通过 openai SDK 调用）
- Rerank：bge-reranker-v2-m3（通过 sentence-transformers 或 xinference）
- 工具协议：原生 Function Calling（@tool + bind_tools）
- 状态持久化：SQLite Checkpointer（langgraph-checkpoint-sqlite）
- 配置：python-dotenv + pydantic-settings

## 项目结构（严格按此创建）
personal-knowledge-agent/
├── .env.example
├── requirements.txt
├── config.py
├── main.py
├── prompts/
│   └── system_prompt.py
├── tools/
│   ├── __init__.py
│   ├── knowledge_search.py
│   ├── list_documents.py
│   └── create_note.py
├── rag/
│   ├── __init__.py
│   ├── loader.py
│   ├── splitter.py
│   ├── embedder.py
│   ├── reranker.py
│   ├── retriever.py
│   └── vectorstore.py
└── graph/
    ├── __init__.py
    ├── state.py
    ├── nodes.py
    ├── edges.py
    └── builder.py

## 关键实现规则

### GLM-5.2 相关
1. 使用 `openai.OpenAI` 客户端，`base_url="https://open.bigmodel.cn/api/paas/v4"`
2. 模型名：`"glm-5.2"`
3. GLM-5.2 默认开启 Thinking → 系统提示词中不得包含推理步骤类指令（不要 "think step by step"，不要 ReAct 模板）。只写规则、约束和输出格式。
4. 对于简单的路由/分类节点，传入额外参数 `extra_body={"thinking": {"type": "disabled"}}` 以节省 token
5. 在适用场景下通过 tool_stream 参数支持流式输出
6. 处理 GLM 专有错误码（1301=余额不足，1305=触发限流），带重试逻辑

### LangGraph 相关
1. 定义 TypedDict 状态，包含：messages（Annotated[list, add_messages]）、retrieved_docs（list[dict]）、current_query（str）、retry_count（int）、final_answer（str | None）
2. 图流程：START → intent_router → [knowledge_search_node | direct_response_node] → rerank_node → generate_node → END
3. 在 generate_node 之后加条件边：若答案质量检查不通过 且 retry_count < 1 → rewrite_query_node → knowledge_search_node；否则 → END
4. 使用 SqliteSaver 作为 checkpointer，初始化时支持 thread_id
5. 所有节点必须兼容 async

### RAG 流水线相关
1. 切分器：先 MarkdownHeaderTextSplitter，再以 RecursiveCharacterTextSplitter(chunk_size=512, overlap=64) 作为兜底
2. Embedding：使用 openai SDK 的 embeddings.create()，model="embedding-3"，batch size=20
3. Retriever：ChromaDB similarity_search_with_relevance_scores，top_k=10
4. Reranker：对 top-10 做 cross-encoder 重排 → 返回 top-5。若重排器不可用，优雅降级为原始检索结果
5. 每个片段存储的元数据：source_doc、section_header、chunk_index、created_at

### 工具实现相关
1. knowledge_search：入参 query(str) + 可选 filters(dict)。返回 {content, metadata, score} 列表。必须在内部调用 retriever。
2. list_documents：从 ChromaDB 元数据聚合返回 {doc_name, tags, updated_at} 列表
3. create_note：入参 title(str) + content(str)。保存为新的 markdown 文件到 ./data/notes/，并立即建立 ChromaDB 索引
4. 所有工具都用 @tool 装饰器，并写完整 docstring（GLM-5.2 高度依赖 docstring 来理解工具）
5. 工具 schema 必须兼容 openai function calling 格式

### 系统提示词规则
1. 存放在 prompts/system_prompt.py 的 SYSTEM_PROMPT 常量中
2. 内容仅聚焦：角色定义、绝对规则（知识库优先、零幻觉、强制引用）、工具使用规则、回答格式
3. 不要推理协议，不要 chain-of-thought 指令（思考由 Thinking 承担）
4. 引用格式：[来源: 文档名, 章节/页码]
5. 语言：中文指令，英文技术术语保留

### 错误处理
1. API 失败：指数退避重试（最多 3 次）
2. 检索为空：自动改写查询一次，之后明确返回"未找到"
3. 重排器失败：降级为原始分数，打印警告日志
4. 工具执行错误：向 LLM 返回结构化错误信息，由它决定下一步
5. 绝不静默吞掉异常

## 交付物
按项目结构生成**全部**文件，代码必须完整、可运行。不允许占位符，不允许 "// TODO"，不允许 "..."。每个文件都要达到生产可用标准。

生成完所有文件后，还需提供：
1. 包含全部必需变量的 .env.example
2. 安装配置命令（pip install、chromadb 初始化等）
3. 一个加载示例 markdown 文件并执行 3 条测试查询的测试脚本
4. 已知限制与下一步建议

## 现在开始
先创建 requirements.txt 和 config.py，然后按依赖顺序逐个文件推进。不要提澄清问题 —— 直接按本规格实现。若某个决策存在歧义，选择最稳妥的方案，并在代码注释中记录说明。
