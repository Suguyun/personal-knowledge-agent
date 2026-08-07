# Task: Implement Personal Knowledge Assistant Agent

## Tech Stack (LOCKED - Do NOT change)
- LLM: GLM-5.2 via OpenAI-compatible API (base_url: https://open.bigmodel.cn/api/paas/v4)
- Framework: LangGraph (Python, StateGraph)
- Vector DB: ChromaDB (local persistent)
- Embedding: zhipuai embedding-3 (via openai SDK)
- Rerank: bge-reranker-v2-m3 (via sentence-transformers or xinference)
- Tool Protocol: Native Function Calling (@tool + bind_tools)
- State Persistence: SQLite Checkpointer (langgraph-checkpoint-sqlite)
- Config: python-dotenv + pydantic-settings

## Project Structure (CREATE EXACTLY)
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

## CRITICAL IMPLEMENTATION RULES

### GLM-5.2 Specifics
1. Use `openai.OpenAI` client with `base_url="https://open.bigmodel.cn/api/paas/v4"`
2. Model name: `"glm-5.2"`
3. GLM-5.2 has Thinking ENABLED BY DEFAULT → System Prompt must NOT contain reasoning step instructions (no "think step by step", no ReAct template). Only rules, constraints, and format.
4. For simple routing/classification nodes, pass extra_body={"thinking": {"type": "disabled"}} to save tokens
5. Support streaming with tool_stream parameter where applicable
6. Handle GLM-specific error codes (1301=insufficient balance, 1305=rate limit) with retry logic

### LangGraph Specifics
1. Define TypedDict State with: messages (Annotated[list, add_messages]), retrieved_docs (list[dict]), current_query (str), retry_count (int), final_answer (str | None)
2. Graph flow: START → intent_router → [knowledge_search_node | direct_response_node] → rerank_node → generate_node → END
3. Add conditional edge after generate_node: if answer quality check fails AND retry_count < 1 → rewrite_query_node → knowledge_search_node; else → END
4. Use SqliteSaver for checkpointer, initialize with thread_id support
5. All nodes must be async-compatible

### RAG Pipeline Specifics
1. Splitter: MarkdownHeaderTextSplitter first, then RecursiveCharacterTextSplitter(chunk_size=512, overlap=64) as fallback
2. Embedding: Use openai SDK embeddings.create() with model="embedding-3", batch size=20
3. Retriever: ChromaDB similarity_search_with_relevance_scores, top_k=10
4. Reranker: Cross-encoder rerank on top-10 → return top-5. If reranker unavailable, gracefully degrade to raw retrieval results
5. Metadata stored per chunk: source_doc, section_header, chunk_index, created_at

### Tool Implementation Specifics
1. knowledge_search: Takes query(str) + optional filters(dict). Returns list of {content, metadata, score}. Must call retriever internally.
2. list_documents: Returns list of {doc_name, tags, updated_at} from ChromaDB metadata aggregation
3. create_note: Takes title(str) + content(str). Saves as new markdown file to ./data/notes/ AND indexes into ChromaDB immediately
4. All tools use @tool decorator with complete docstring (GLM-5.2 relies heavily on docstring for tool understanding)
5. Tool schemas must be compatible with openai function calling format

### System Prompt Rules
1. Store in prompts/system_prompt.py as SYSTEM_PROMPT constant
2. Content focuses ONLY on: role definition, absolute rules (knowledge-base-first, zero-hallucination, mandatory citation), tool usage rules, response format
3. NO reasoning protocol, NO chain-of-thought instructions (Thinking handles this)
4. Citation format: [来源: 文档名, 章节/页码]
5. Language: Chinese instructions, English technical terms preserved

### Error Handling
1. API failures: Exponential backoff retry (max 3 attempts)
2. Empty retrieval: Auto-rewrite query once, then explicit "not found" response
3. Reranker failure: Fallback to raw scores, log warning
4. Tool execution error: Return structured error message to LLM, let it decide next action
5. Never silently swallow exceptions

## DELIVERABLES
Generate ALL files listed in Project Structure with COMPLETE, RUNNABLE code. No placeholders, no "// TODO", no "...". Every file must be production-ready.

After generating all files, provide:
1. .env.example with all required variables
2. Setup commands (pip install, chromadb init, etc.)
3. A test script that loads sample markdown files and runs 3 test queries
4. Known limitations and next-step recommendations

## START NOW
Begin by creating requirements.txt and config.py, then proceed file-by-file in dependency order. Do not ask clarifying questions — implement based on these specifications. If a decision is ambiguous, choose the most robust option and document it in code comments.