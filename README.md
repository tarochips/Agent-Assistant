# Agent-ready RAG V4

这是一个基于 FastAPI、PostgreSQL 和 pgvector 的中文/英文知识库问答项目。V3 提供混合检索与重排，V4 使用 LangGraph 编排有状态 Agent，支持受控工具调用、工具循环、PostgreSQL Checkpoint 和人工确认后恢复执行。

## 六阶段路线

| 版本 | 目标 |
|---|---|
| V1 | 模块化 FastAPI、流式 SSE、基础 RAG |
| V2 | PostgreSQL、pgvector、文档和会话持久化 |
| V3 | 语义 Embedding、BM25、RRF、重排和检索评估 |
| V4（当前） | LangGraph 单 Agent、受控工具调用、分支循环、Checkpoint 和中断恢复 |
| V5 | 有证据引用的并行研究、审查和报告生成 |
| V6 | 任务队列、观测、效果评估、安全和 MCP 接入 |

当前范围是 V4。V5 及之后的多 Agent 研究和生产化能力尚未实现。

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
