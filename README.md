# Agent-ready RAG V6（开发中）

这是一个基于 FastAPI、PostgreSQL 和 pgvector 的中文/英文知识库问答项目。支持混合检索、带人工确认的 Agent 工具调用，以及并行研究、证据审查和带来源的报告生成。

## 六阶段路线

| 版本 | 目标 |
|---|---|
| V1 | 模块化 FastAPI、流式 SSE、基础 RAG |
| V2 | PostgreSQL、pgvector、文档和会话持久化 |
| V3 | 语义 Embedding、BM25、RRF、重排和检索评估 |
| V4 | LangGraph 单 Agent、受控工具调用、分支循环、Checkpoint 和中断恢复 |
| V5 | 有证据引用的并行研究、审查和报告生成 |
| V6（开发中） | 持久化任务队列、基础观测、评估、安全和 MCP 接入 |

V6 基础能力已经接入；仍需在真实 PostgreSQL、模型和多工作进程下完成端到端验证。

## V6 后台研究任务

升级数据库后，在两个终端分别启动 Web 服务和独立的任务工作进程：

    python -m alembic upgrade head
    python run.py
    python -m app.jobs.worker

通过 `POST /api/research-jobs` 提交 `{"question":"研究问题"}`，获取任务 ID；
`GET /api/research-jobs/{id}` 查看状态和结果，`GET /api/research-jobs` 查看最近任务，
`POST /api/research-jobs/{id}/cancel` 取消任务，`GET /api/research-jobs/metrics` 查看各状态数量。
任务结果保存在任务表，不会自动写入聊天会话；现有页面的深度研究仍使用同步接口。

任务由 PostgreSQL 行锁和租约分配，工作进程续租；执行失败或租约过期可重新领取，最多尝试
`JOB_MAX_ATTEMPTS` 次。完成写入需要持有有效租约，避免旧工作进程覆盖结果。
取消后工作进程在下次续租时停止，期间已经发出的模型调用可能仍会产生费用；重试也可能重复调用模型。
`JOB_LEASE_SECONDS`、`JOB_POLL_SECONDS` 和 `JOB_MAX_ATTEMPTS` 可配置。

## 访问控制和 MCP

默认监听 `127.0.0.1`。可设置 `API_ACCESS_TOKEN`，所有 `/api/` 接口（存活检查除外）需要
`Authorization: Bearer <令牌>`；页面可通过“设置访问令牌”在当前标签页使用令牌。
未设置令牌时 API 不做身份验证，现有 `.env` 中的 `APP_HOST` 会覆盖默认监听地址。
这个单令牌方案没有用户隔离、限流或 TLS，不能直接视作公网部署方案。

运行 `python -m app.mcp_server.server` 启动本地 stdio MCP 服务，向 MCP 客户端提供
`search_knowledge` 和 `list_documents` 两个只读工具。MCP 进程使用本机数据库配置和权限，
应仅由可信的本地客户端启动；不提供远程 MCP 监听或删除工具。

## 报告引用评估

`python -m app.evaluation.reports <标注文件.json>` 计算引用编号有效率和人工标注证据命中率。
文件格式示例：

```json
[{"report": "结论 [S1]", "citations": [{"chunk_id": "doc123:0"}], "relevant_chunk_ids": ["doc123:0"]}]
```

这些指标只检查引用结构和证据 ID，不评判结论事实正确性；需要用真实数据和人工标注评估报告质量。

## 深度研究

在页面勾选“深度研究”，或向 `POST /api/chat/stream` 发送 `mode: "research"`。
规划员拆分问题，每个研究员运行独立的 LangGraph 子图，保存自己的消息、检索记录和证据，
并可在限定轮次内自主调用只读知识库检索工具。研究员并行工作，报告撰写员依据去重后的资料生成草稿；
审查员返回结构化结论，决定通过、要求修订或发起补充研究。修改或补充后会重新审查，达到上限仍未通过时报告会明确标记。
无检索结果时返回资料不足提示。
报告和来源保存至普通会话，页面会在整个研究完成后收到报告；普通聊天保持原有流式和确认机制。

可通过 `RESEARCH_MAX_TASKS`（默认 3）、`RESEARCH_MAX_SUPPLEMENTAL`（默认 2）、
`RESEARCH_WORKER_MAX_TOOL_ROUNDS`（每个研究员默认 2）、`RESEARCH_MAX_REVIEW_ROUNDS`（默认 2）和
`RESEARCH_TIMEOUT_SECONDS`（单次模型调用或检索超时，默认 120 秒）约束执行。
研究只使用本地知识库，不会联网搜索；引用编号会校验，但模型生成的结论与证据是否一致仍需人工核验。
研究流程不使用聊天 Agent 的 checkpoint，中途失败需要重新提交。会话内的研究使用当前会话摘要和近期消息理解问题；独立后台任务没有关联会话，仅使用任务问题。
规划、研究、审查和写作角色目前共享同一模型服务；子 Agent 状态在单次研究执行期间隔离，不持久化。

## 对话上下文管理

普通问答和会话内深度研究使用“较早对话摘要 + 近期原始对话 + 本次检索证据”。
旧对话按批次调用模型增量合并到会话摘要，原始消息不会删除；摘要和已覆盖消息数存入数据库。
近期消息受到 `HISTORY_MAX_MESSAGES`（默认 20 条）和 `CONTEXT_RECENT_MAX_CHARS`（默认 16000 字符）共同限制，尽量保留完整问答轮次。
`CONTEXT_SUMMARY_BATCH_MESSAGES`、`CONTEXT_SUMMARY_BATCH_CHARS`、`CONTEXT_SUMMARY_MAX_CHARS` 和 `CONTEXT_SUMMARY_TIMEOUT_SECONDS` 可控制摘要批次、输出和超时。
这些字符限制不是精确的模型 token 预算，也不限制检索材料或工具循环的总 token。

摘要通过条件更新防止并发覆盖；模型失败、摘要过长、单条历史超过批次限制或摘要并发冲突时会返回错误，重试后可继续，原始历史仍保留。
摘要可能遗漏或误述内容，它只是理解对话的辅助背景，不能替代知识库证据或授权工具操作；增量摘要需要额外调用模型并发送待压缩的历史对话。
普通问答仍使用当前问题直接检索，没有自动改写上下文相关的追问。数据库升级需要执行 `python -m alembic upgrade head`。

## V3 检索流程

    文档分块
      -> FastEmbed multilingual E5 small（本地 ONNX，384 维）
      -> PostgreSQL / pgvector

    用户问题
      -> 语义向量候选 ─┐
      -> jieba + BM25 ──┴-> RRF -> 可选 BGE Cross-Encoder -> Top K 引用

V3 选用 intfloat/multilingual-e5-small 生成 384 维向量，以保持现有 pgvector 列维度；查询和文档分别按 E5 约定添加 query:、passage: 前缀。BM25 使用 jieba 对中文分词，并保留英文和数字词项。RRF 合并两路排序。默认使用本地 BAAI/bge-reranker-base 对有限候选集重排；不希望下载重排模型时可设置 RETRIEVAL_RERANKER_ENABLED=false，此时以 RRF 排序返回。

模型缓存位于 DATA_DIR/model_cache。默认 DATA_DIR 是项目的 data 目录，因此权重会写入 E 盘项目目录，不使用 Docker 卷，也不会默认放在用户的 C 盘缓存。首次上传或查询需要从模型仓库下载权重；Embedding 模型和重排模型会分别在首次使用时加载。重排模型体积约 1 GB，请确保 E 盘有足够空间和可用网络。

## V4 LangGraph 工作流

    START -> retrieve -> agent --回答--> END
                              |工具调用
                              v
                            tools -> agent（循环）
                              |
                              +--删除文档 -> interrupt（等待用户确认）
                                                   |
                                           Command(resume=...) -> 恢复

Agent 可调用知识库检索、文档列表和删除文档三个工具。工具参数由 Pydantic 模型生成 JSON Schema，并在执行前进行本地校验；删除文档会通过 `interrupt()` 暂停，用户确认后用相同的会话 ID 恢复。工具调用次数受配置限制。

LangGraph Checkpoint 使用与业务数据相同的 PostgreSQL 数据库，但由 `AsyncPostgresSaver.setup()` 管理独立的 checkpoint 表。首次启动应用时会自动创建或升级这些表。普通会话消息仍写入业务 `sessions/messages` 表；Checkpoint 保存 Agent 工作流状态，供同一 `thread_id` 继续执行。

学习文档逐文件解释 StateGraph、节点、条件分支、工具循环、checkpoint、`interrupt()` 和恢复 API：[docs/LANGGRAPH_LEARNING.md](docs/LANGGRAPH_LEARNING.md)。

## 安装与运行

需要 Python 3.11+、Docker Compose、可用的 DeepSeek API Key。项目运行时只需要启动 PostgreSQL 容器，FastAPI 由 Python 运行。

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install -e ".[dev]"
    Copy-Item .env.example .env

在 .env 中填写 `DEEPSEEK_API_KEY`，然后启动数据库并升级业务表结构：

    docker compose up -d db
    python -m alembic upgrade head

LangGraph checkpoint 表会在应用第一次启动时由 saver 自动创建，无需额外 Alembic revision。

V2 已有知识库需要用新的语义模型重算已有向量。备份数据库后运行：

    python -m app.retrieval.rebuild_embeddings

这个命令按批次重算现有分块，运行中断后可重复执行。完成后启动应用：

    python run.py

- 应用：http://127.0.0.1:8000
- Swagger：http://127.0.0.1:8000/docs
- 数据库健康检查：http://127.0.0.1:8000/api/health/ready

新上传的文档会自动使用当前配置的 Embedding 模型。数据库里若仍有旧模型生成的向量，检索会提示先执行重建命令，避免把不同模型空间的向量混在同一次搜索中。

## 检索评估

准备一个 JSON 文件，按数据库的 chunk_id（文档 ID:分块序号）标注相关分块：

    [
      {
        "query": "PostgreSQL 如何存储向量？",
        "relevant_chunk_ids": ["doc123:0", "doc123:2"]
      }
    ]

运行 python -m app.retrieval.evaluate data/evaluation/retrieval_cases.json --k 5 会输出 Recall@K、MRR@K 和 Hit Rate@K。使用真实问题和人工标注的相关分块，比较调整前后的检索效果。

## 开发检查

    ruff format --check .
    ruff check .
    mypy app tests
    pytest

测试采用假的 Embedding、重排器和内存仓库，不需要启动 Docker，也不会请求 DeepSeek。
要执行 PostgreSQL 任务租约集成测试，可将 `TEST_DATABASE_URL` 指向专用测试数据库
（使用 `postgresql+asyncpg://` 格式）后运行 `pytest tests/integration/test_postgres_jobs.py`。
测试会创建并删除一个随机命名的临时 schema；未配置时自动跳过。

## 目录说明

- app/api/：HTTP 接口、依赖注入和健康检查。
- app/agent/：LangGraph 状态、工作流图、工具参数 Schema 和工具执行。
- app/services/：聊天、文档和会话用例。
- app/retrieval/：切块、语义向量、BM25、RRF、重排和评估。
- app/repositories/postgres.py：PostgreSQL 文档和会话读写。
- app/db/models.py：SQLAlchemy 数据模型。
- migrations/：Alembic 数据库迁移。
- docs/V3_DESIGN.md：版本路线和 V3 技术选型、改动记录。
- docs/LANGGRAPH_LEARNING.md：V4 LangGraph 系统学习指南和代码映射。

## 当前边界

- 当前 BM25 索引在每次查询时从 PostgreSQL 读取分块并在进程内评分，适合当前个人项目规模；更大语料应迁移到持久化倒排索引或数据库全文检索方案。
- 重排是可配置步骤；Cross-Encoder 会增加本机模型缓存和 CPU 推理时间。
- 修改 Embedding 模型后需重新生成所有分块的向量。当前数据库仍使用 384 维；换成不同维度模型时，需要新增数据库迁移并重建向量索引。
- 当前上传仍只支持 UTF-8 .txt/.md；当前 Agent 工具只管理知识库检索、文档列表和删除。
