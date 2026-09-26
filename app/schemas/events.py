from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int = Field(ge=0)
    score: float = Field(ge=0.0, le=1.0)
    excerpt: str


class RetrievalEvent(BaseModel):
    type: Literal["retrieval"] = "retrieval"
    session_id: str
    sources: list[Citation]


class TokenEvent(BaseModel):
    type: Literal["token"] = "token"
    delta: str


class CompletedEvent(BaseModel):
    type: Literal["completed"] = "completed"
    session_id: str
    citations: list[Citation]


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str
