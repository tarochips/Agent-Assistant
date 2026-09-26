from pydantic import BaseModel, Field


class DocumentSummary(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int = Field(ge=0)
    uploaded_at: str


class DocumentUploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int = Field(ge=0)


class DeleteResponse(BaseModel):
    deleted: bool
