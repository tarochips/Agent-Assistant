from pydantic import BaseModel, ConfigDict, Field


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchKnowledgeArguments(ToolArguments):
    query: str = Field(min_length=1, max_length=4000, description="要检索的知识库问题")
    top_k: int = Field(default=4, ge=1, le=10, description="最多返回的分块数量")


class ListDocumentsArguments(ToolArguments):
    """No arguments; the tool lists the documents currently in the knowledge base."""


class DeleteDocumentArguments(ToolArguments):
    document_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="要删除的文档 ID",
    )
