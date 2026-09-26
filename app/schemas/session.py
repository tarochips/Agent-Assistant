from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.schemas.events import Citation


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    sources: list[Citation] = Field(default_factory=list)


class SessionDetail(BaseModel):
    id: str
    title: str
    created_at: str
    messages: list[ChatMessage] = Field(default_factory=list)


class SessionSummary(BaseModel):
    id: str
    title: str
    created_at: str
    message_count: int = Field(ge=0)


class CreateSessionRequest(BaseModel):
    title: str | None = Field(default=None, max_length=100)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None
