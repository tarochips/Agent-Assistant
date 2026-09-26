from typing import Any

from app.core.exceptions import NotFoundError
from app.repositories.sessions import JsonSessionRepository


class SessionService:
    def __init__(self, repository: JsonSessionRepository) -> None:
        self._repository = repository

    def create(self, title: str | None = None) -> dict[str, Any]:
        return self._repository.create(title)

    def get(self, session_id: str) -> dict[str, Any]:
        session = self._repository.get(session_id)
        if session is None:
            raise NotFoundError("session not found", code="SESSION_NOT_FOUND")
        return session

    def list(self) -> list[dict[str, Any]]:
        return self._repository.list_sessions()

    def delete(self, session_id: str) -> None:
        if not self._repository.delete(session_id):
            raise NotFoundError("session not found", code="SESSION_NOT_FOUND")
