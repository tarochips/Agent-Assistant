import json
from dataclasses import dataclass, field
from typing import Any

from langgraph.types import interrupt
from pydantic import BaseModel, ValidationError

from app.agent.schemas import (
    DeleteDocumentArguments,
    ListDocumentsArguments,
    SearchKnowledgeArguments,
)
from app.services._async_utils import resolve


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    arguments: type[BaseModel]


@dataclass(slots=True)
class ToolResult:
    content: str
    context: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)


TOOL_DEFINITIONS = (
    ToolDefinition(
        "search_knowledge",
        "在项目知识库中检索问题相关的文档片段。回答需要证据或需要补充检索时使用。",
        SearchKnowledgeArguments,
    ),
    ToolDefinition(
        "list_documents",
        "列出当前知识库中的文档及分块数量。",
        ListDocumentsArguments,
    ),
    ToolDefinition(
        "delete_document",
        "删除指定知识库文档。仅当用户明确要求删除时调用；执行前必须等待用户确认。",
        DeleteDocumentArguments,
    ),
)


def openai_tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": definition.name,
                "description": definition.description,
                "parameters": definition.arguments.model_json_schema(),
            },
        }
        for definition in TOOL_DEFINITIONS
    ]


async def execute_tool(
    name: str,
    raw_arguments: str,
    *,
    retriever: Any,
    document_repository: Any,
    settings: Any,
) -> ToolResult:
    definition = next((tool for tool in TOOL_DEFINITIONS if tool.name == name), None)
    if definition is None:
        return ToolResult(f"未知工具：{name}")

    try:
        arguments = definition.arguments.model_validate_json(raw_arguments)
    except ValidationError as exc:
        return ToolResult(f"工具参数无效：{exc.errors(include_url=False)}")

    if isinstance(arguments, SearchKnowledgeArguments):
        results = await resolve(
            retriever.search(
                arguments.query,
                top_k=arguments.top_k,
                min_score=settings.retrieval_min_score,
            )
        )
        context = [
            {
                "chunk_id": result.chunk_id,
                "document_id": result.document_id,
                "filename": result.filename,
                "chunk_index": result.chunk_index,
                "text": result.text,
                "score": result.score,
            }
            for result in results
        ]
        citations = [result.to_citation().model_dump(mode="json") for result in results]
        if not results:
            return ToolResult("知识库中没有找到相关资料。")
        formatted = [
            f"[{index}] {item['filename']} · 分块 {item['chunk_index'] + 1}\n{item['text']}"
            for index, item in enumerate(context, start=1)
        ]
        return ToolResult("\n\n".join(formatted), context=context, citations=citations)

    if isinstance(arguments, ListDocumentsArguments):
        documents = await resolve(document_repository.list_documents())
        return ToolResult(json.dumps(documents, ensure_ascii=False, default=str))

    if isinstance(arguments, DeleteDocumentArguments):
        decision = interrupt(
            {
                "action": "delete_document",
                "document_id": arguments.document_id,
                "message": f"Agent 请求删除文档 {arguments.document_id}，是否允许？",
            }
        )
        if not isinstance(decision, dict) or decision.get("approved") is not True:
            return ToolResult("用户拒绝了删除操作；文档没有被删除。")
        deleted = await resolve(retriever.remove_document(arguments.document_id))
        return ToolResult("文档已删除。" if deleted else "未找到该文档，未执行删除。")

    return ToolResult("工具没有可执行的处理器。")
