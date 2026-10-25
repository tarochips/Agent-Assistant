# V3 设计与实现记录

更新日期：2026-10-25  
状态：实现和自动化检查完成；尚未连接真实 PostgreSQL 或下载模型权重做端到端推理。

## 1. 项目六阶段路线

| 版本 | 名称 | 交付范围 |
|---|---|---|
| V1 | Agent-ready RAG 基础 | 模块化 FastAPI、基础 RAG、来源引用和 SSE 流式接口 |
| V2 | 持久化知识库 | PostgreSQL、pgvector、文档/分块/会话持久化 |
| V3 | 高质量检索（当前） | 语义向量 + BM25 混合检索、RRF、Cross-Encoder 重排、离线检索评估 |
| V4（当前） | 单 Agent | LangGraph 工作流、受控工具调用、结构化参数和 Checkpoint |
| V5 | 深度研究 Agent | 并行研究、证据审查、报告生成和有界执行 |
| V6 | 生产化 | 任务队列、观测、评估、安全、MCP 接入 |

V3 本身不引入 Agent 工作流；V4 已使用 LangGraph。LangChain 的整套 Agent 抽象和 LlamaIndex 均未引入；当前 PostgreSQL/pgvector 数据和 V3 检索代码继续由本项目管理。V4 实现和学习资料见 [LANGGRAPH_LEARNING.md](LANGGRAPH_LEARNING.md)。

## 2. V3 目标与边界

V2 的 pgvector 中存储的是 384 维字符哈希向量，适合词面重合检索，不具备语义 Embedding 的同义改写召回能力。V3 的目标是把检索变为可测试、可观察的两阶段流程：

1. 分别用稠密语义向量和 BM25 找到候选分块。
2. 用 RRF 合并两路排序。
3. 用 Cross-Encoder 对有限候选集重排。
4. 使用 Recall@K、MRR@K、Hit Rate@K 对标注问题集计算检索质量。

当前仍支持 UTF-8 TXT/Markdown 上传、FastAPI、PostgreSQL、pgvector 和 DeepSeek 对话。PDF/Word 解析、LangGraph Agent、JSON Schema 工具调用和生产任务队列不属于 V3。

## 3. 技术选型

| 组件 | 选型 | 原因 |
|---|---|---|
| Embedding 推理 | FastEmbed + ONNX Runtime | 本地 CPU 推理，不把文档文本发给额外 Embedding API；FastEmbed 支持通过 cache_dir 指定模型缓存位置。 |
| Embedding 模型 | intfloat/multilingual-e5-small，384 维 | 支持多语言检索；384 维与现有 pgvector 列一致。查询使用 query: 前缀，文档使用 passage: 前缀。 |
| 关键词检索 | 自实现 BM25 + jieba 分词 | 中文需要词语切分；依赖较轻，分数计算和参数可单元测试。 |
| 融合排序 | Reciprocal Rank Fusion（RRF） | 合并不同范围、不同量纲的排名分数，不需要在 V3 先训练融合模型。 |
| 重排 | FastEmbed TextCrossEncoder + BAAI/bge-reranker-base | 对 RRF 候选执行 query-document 相关性打分；模型支持中文和英文，使用 MIT 许可。 |
| 持久化 | PostgreSQL + pgvector | 沿用 V2，保留 HNSW cosine 查询和现有 384 维向量列。 |
| 评估 | Recall@K、MRR@K、Hit Rate@K | 以人工标注的相关 chunk ID 计算结果，便于比较不同检索参数。 |
| 框架 | V3 不引入 LangChain、LlamaIndex 或 LangGraph | 本版聚焦检索算法和评估；LangGraph 留给 V4 的有状态 Agent 工作流。 |

FastEmbed 的 API 支持显式传入模型 cache_dir；其模型清单将 BAAI/bge-reranker-base 列为约 1.04 GB。官方资料：[FastEmbed 使用和缓存 API](https://github.com/qdrant/fastembed)、[FastEmbed 模型清单](https://qdrant.github.io/fastembed/examples/Supported_Models/)、[BGE reranker 模型说明和语言/许可信息](https://huggingface.co/BAAI/bge-reranker-base)。

## 4. 检索和数据流程

    上传文本
      -> 按字符窗口分块
      -> E5 passage Embedding
      -> PostgreSQL chunks.embedding（384 维）
      -> chunks.embedding_model 记录模型版本

    用户问题
      -> E5 query Embedding -> pgvector cosine Top-N ─┐
      -> jieba + BM25 全量分块 Top-N ────────────────┴-> RRF Top-N
      -> 可选 BGE Cross-Encoder 重排
      -> 返回 Top-K SearchResult 和来源引用

RRF 常数默认 60，候选数默认 30，最终返回数沿用 RETRIEVAL_TOP_K。重排分数经过 sigmoid 转为 0 到 1 的展示分数；没有启用重排时，返回归一化后的 RRF 分数。

当前 BM25 查询会从 PostgreSQL 拉取所有 chunk 并在应用进程中计算，复杂度随语料规模线性增加。这适合个人项目和小型知识库；语料扩大后应使用持久化倒排索引或专用搜索后端。

## 5. 数据库迁移与旧向量重建

V3 保持向量列 384 维，不需要改写现有 HNSW 索引。Alembic revision 0002 为 chunks 增加 embedding_model 字段；迁移时已有行会标记为 legacy-char-hash-384-v1。新文档会写入当前模型 ID。

从 V2 升级时按顺序执行：

1. 备份 PostgreSQL 数据库。
2. 执行 python -m alembic upgrade head。
3. 执行 python -m app.retrieval.rebuild_embeddings。
4. 检查命令输出的更新 chunk 数，再启动 API。

重建命令以每批 64 个分块重新计算向量并更新模型标记。若中途停止，可以再次运行。检索时会检测是否仍存在旧模型向量；检测到混合模型数据时，会提示先执行重建，避免把不同向量空间的距离放进同一次排序。

FastEmbed 的 Embedding 和重排模型都使用 DATA_DIR/model_cache。默认 DATA_DIR 位于项目目录，因此模型缓存和数据库卷都在项目 E 盘下。首次使用需要网络下载模型；重排默认开启并会额外下载约 1 GB 权重。可将 RETRIEVAL_RERANKER_ENABLED 设为 false 只用 BM25 + 向量 + RRF。

## 6. 检索评估

评估输入为 JSON 数组，每项包括 query 和非空的 relevant_chunk_ids，例如：

    [
      {
        "query": "PostgreSQL 如何存储向量？",
        "relevant_chunk_ids": ["doc123:0", "doc123:2"]
      }
    ]

命令：

    python -m app.retrieval.evaluate data/evaluation/retrieval_cases.json --k 5

chunk ID 格式是文档 ID 加冒号加从 0 开始的 chunk 序号。数据集由开发者根据实际知识库和人工判断维护。评估命令会通过当前配置的混合检索器逐条查询，并输出 Recall@K、MRR@K、Hit Rate@K 和样本数。

本仓库单元测试验证指标计算与检索各阶段；要评价真实召回率，仍需准备有代表性的查询和人工标签。

## 7. 改动和新增记录

### 修改

- README.md：说明 V1-V6 路线、V3 启动、向量重建、检索评估和适用边界。
- .env.example：增加 Embedding、BM25/RRF 候选数、重排模型和线程配置。
- .gitignore：排除 data/model_cache，避免模型文件进入 Git。
- pyproject.toml、requirements.txt：增加 fastembed、jieba；开发依赖增加 mypy。
- app/core/config.py：增加 V3 配置和模型缓存目录。
- app/db/models.py：为每个 chunk 保存 embedding_model。
- app/repositories/postgres.py：增加 chunk 全量读取、旧模型检测和批量向量更新。
- app/retrieval/vectorizer.py：增加本地 E5 适配器，保留旧哈希向量器用于兼容/测试。
- app/retrieval/vector.py：编排两路候选、RRF、重排和旧向量重建。
- app/api/dependencies.py、app/api/health.py：接入 V3 默认模型并展示检索配置。
- app/main.py：API 版本提升到 3.0.0。
- app/llm/client.py：修正流式客户端 Protocol 和 OpenAI 类型声明，不改变运行逻辑。
- app/repositories/sessions.py、app/retrieval/tfidf.py：补齐类型注解以通过全项目类型检查。
- 其他被 Ruff 格式化的 V2 文件：只调整格式，不改变检索或 API 行为。
- tests/unit/test_vector_retriever.py：升级向量检索测试，覆盖混合检索、旧向量检测和批量重建。

### 新增

- migrations/versions/0002_embedding_model.py：增加向量模型版本字段。
- app/retrieval/bm25.py：中文/英文 BM25 关键词检索。
- app/retrieval/fusion.py：RRF 排名融合。
- app/retrieval/reranker.py：本地 Cross-Encoder 适配器和关闭重排时的实现。
- app/retrieval/rebuild_embeddings.py：旧向量批量重建 CLI。
- app/retrieval/evaluation.py：检索指标、数据集验证和评估 CLI。
- tests/unit/test_bm25.py、tests/unit/test_fusion.py、tests/unit/test_reranker.py、tests/unit/test_retrieval_evaluation.py：对应阶段和指标的单元测试。

## 8. 验证记录

| 命令 | 结果 |
|---|---|
| 安装 FastEmbed、jieba、mypy | 成功：fastembed 0.9.1、jieba 0.42.1、mypy 2.4.0、onnxruntime 1.31.0 |
| ruff format --check . | 通过：62 个文件均已格式化 |
| ruff check . | 通过：All checks passed |
| mypy app tests | 通过：56 个源文件无类型错误 |
| pytest | 通过：28 passed（2.94 秒） |
| alembic history | 通过：0001_postgres_pgvector -> 0002_embedding_model（head） |
| git diff --check | 通过：无空白错误 |

本轮基础测试不连接 PostgreSQL 或 DeepSeek，也不下载模型权重。数据库迁移和真实模型检索需要在 PostgreSQL 可用且模型首次下载完成后另行执行。
