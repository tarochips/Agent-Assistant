# Agent-ready RAG V1

这是一个可追溯、可测试、可扩展的轻量级 RAG 知识库。V1 仍使用字符级
TF-IDF 和本地 JSON，以较小改动建立未来接入 PostgreSQL、pgvector 和
LangGraph 所需的模块边界。

## V1 能力

- FastAPI API 与自动 OpenAPI 文档
- UTF-8 `.txt`、`.md` 文档上传
- 字符级 TF-IDF 检索和可配置相关性阈值
- 带文件、分块、相关度和摘要的来源引用
- DeepSeek 兼容接口流式输出
- 类型化 SSE 事件
- JSON 会话和文档数据向后兼容
- 单元测试与接口测试

## 架构

```text
API Route
   -> Service
      -> Retriever
      -> LLM Client
      -> Repository
```

- `app/api/`：处理 HTTP 和依赖注入，不承载业务流程。
- `app/services/`：聊天、文档和会话用例。
- `app/retrieval/`：分块、TF-IDF 索引和检索。
- `app/llm/`：模型客户端适配。
- `app/repositories/`：JSON 持久化。
- `app/schemas/`：请求、响应和 SSE 数据契约。
- `app/core/`：配置、异常和日志。

## 安装

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

在 `.env` 中配置 `DEEPSEEK_API_KEY`。不要提交真实密钥。

## 启动

```powershell
python run.py
```

- 应用：`http://127.0.0.1:8000`
- Swagger：`http://127.0.0.1:8000/docs`
- OpenAPI：`http://127.0.0.1:8000/openapi.json`
- 健康检查：`http://127.0.0.1:8000/api/health/ready`

## SSE 事件

聊天接口 `POST /api/chat/stream` 依次发送：

```text
retrieval -> token... -> completed
```

失败时发送 `error` 事件。可以用以下命令观察原始数据流：

```powershell
curl.exe -N -X POST http://127.0.0.1:8000/api/chat/stream `
  -H "Content-Type: application/json" `
  -d '{"message":"知识库讲了什么？"}'
```

## 测试

```powershell
pytest
ruff check .
```

测试使用假的 LLM 客户端，不会调用真实模型或消耗 API 额度。

## 当前边界

- JSON 适合单机学习项目，不适合多进程并发写入。
- TF-IDF 属于词法检索，不是语义 Embedding。
- 相关性阈值需要在后续评测集上校准。
- V1 暂不支持 PDF、Word、Agent、MinIO 和代码执行沙箱。
