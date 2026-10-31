"""Local read-only MCP server: python -m app.mcp_server.server."""

from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from app.api.dependencies import get_document_repository, get_retriever
from app.core.config import get_settings


def create_server(retriever: Any, repository: Any) -> FastMCP:
    server = FastMCP("RAG Knowledge Base", instructions="只读知识库查询工具。")

    @server.tool()
    async def search_knowledge(
        query: Annotated[str, Field(min_length=1, max_length=4000)],
        top_k: Annotated[int, Field(ge=1, le=10)] = 4,
    ) -> list[dict[str, Any]]:
        """Search knowledge base documents and return passages with source metadata."""
        if not query.strip():
            raise ValueError("query cannot be empty")
        results = await retriever.search(
            query.strip(), top_k=top_k, min_score=get_settings().retrieval_min_score
        )
        return [
            {"citation": item.to_citation().model_dump(mode="json"), "text": item.text}
            for item in results
        ]

    @server.tool()
    async def list_documents() -> list[dict[str, Any]]:
        """List documents available in the knowledge base."""
        return await repository.list_documents()

    return server


def main() -> None:
    create_server(get_retriever(), get_document_repository()).run(transport="stdio")


if __name__ == "__main__":
    main()
