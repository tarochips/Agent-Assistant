# Agent-ready RAG V2

这是一个可追溯、可测试、可扩展的 RAG 知识库。V2 将文档、分块和会话保存到 PostgreSQL，并用 pgvector 执行向量相似度检索。文本向量由本地字符 n-gram 哈希生成，不会把新上传的文档发送到额外的 Embedding 服务。

## 版本范围

- **V1**：FastAPI、UTF-8 `.txt`/`.md` 上传、TF-IDF 检索、JSON 持久化、来源引用、DeepSeek 流式回答。
- **V2**：PostgreSQL 持久化、pgvector cosine 检索、本地固定维度文本向量、Alembic 迁移、V1 JSON 导入工具。

当前向量化是本地字符特征哈希，不是语义 Embedding 模型；它保留 V1 的词面匹配特性，同时由 pgvector 保存并查询向量。要启用语义 Embedding，需要另行选定模型和数据处理服务。

## 架构

```text
API Route
   -> Service
      -> VectorRetriever -> LocalHashVectorizer -> PostgreSQL / pgvector
      -> LLM Client
      -> PostgreSQL Repository
```

- `app/api/`：HTTP 接口和依赖注入。
- `app/services/`：聊天、文档和会话用例。
- `app/retrieval/`：文本切块、向量生成和检索。
- `app/llm/`：DeepSeek 流式聊天客户端。
- `app/repositories/postgres.py`：文档、分块、会话和消息的数据库读写。
- `app/db/models.py`：SQLAlchemy 数据模型。
- `migrations/`：Alembic 数据库迁移。
- `app/repositories/json_io.py`、`documents.py`、`sessions.py`：V1 JSON 导入兼容工具。

## 安装和启动

需要 Python 3.11+、Docker Compose 和一个可用的 DeepSeek API Key。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

首次配置时将 `.env.example` 复制为 `.env`，再填入 `DEEPSEEK_API_KEY`。已有 `.env` 时请保留它，并补充 `DATABASE_URL`。默认本地数据库地址为：

```dotenv
DATABASE_URL=postgresql+asyncpg://rag:rag_dev_password@localhost:5432/agent_rag
```

启动 pgvector 数据库并创建表：

```powershell
docker compose up -d db
python -m alembic upgrade head
```

将现有 V1 JSON 文档和会话导入 PostgreSQL：

```powershell
python -m app.migrations.import_v1_json
```

导入工具可重复运行；已存在的文档和会话会跳过。原始 JSON 文件保持不变，便于回退和核对。确认导入完成后启动应用：

```powershell
python run.py
```

- 应用：`http://127.0.0.1:8000`
- Swagger：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/api/health/ready`

## 检索和聊天

上传时，服务将文本切块，在本地生成 384 维归一化字符 n-gram 向量，再以单个数据库事务保存文档和分块。提问时生成本地查询向量，由 pgvector 按 cosine distance 排序，并保留来源、分块和相关度信息。

聊天接口 `POST /api/chat/stream` 发送 `retrieval`、多个 `token`，最后发送 `completed` 事件；错误时发送 `error` 事件。

```powershell
curl.exe -N -X POST http://127.0.0.1:8000/api/chat/stream `
  -H "Content-Type: application/json" `
  -d '{"message":"知识库讲了什么？"}'
```

## 检查

```powershell
pytest
ruff check .
```

API 测试使用假的 LLM 客户端，不会调用模型服务。

## V1 数据和当前边界

- V1 的 `data/docstore.json` 和 `data/sessions/*.json` 不会被自动删除；先运行导入命令，再核对 PostgreSQL 中的记录。
- 本地哈希向量适合原型和词面检索，对同义改写的召回能力有限。
- 数据库向量列为 384 维；更换向量化算法前，需要迁移列定义并为已有分块重新生成向量。
- PDF/Word 解析、混合检索、重排、LangGraph Agent 和生产部署属于后续版本。
