from typing import Any

from app.core.exceptions import NotFoundError
from app.services._async_utils import resolve


class SessionService:
    def __init__(self, repository: Any) -> None:
        self._repository = repository

    async def create(self, title: str | None = None) -> dict[str, Any]:
        return await resolve(self._repository.create(title))

    async def get(self, session_id: str) -> dict[str, Any]:
        session = await resolve(self._repository.get(session_id))
        if session is None:
            raise NotFoundError("session not found", code="SESSION_NOT_FOUND")
        return session

    async def list(self) -> list[dict[str, Any]]:
        return await resolve(self._repository.list_sessions())

    async def delete(self, session_id: str) -> None:
        if not await resolve(self._repository.delete(session_id)):
            raise NotFoundError("session not found", code="SESSION_NOT_FOUND")
