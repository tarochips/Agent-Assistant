from __future__ import annotations

import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from app.repositories.json_io import JsonFileStore

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class JsonSessionRepository:
    def __init__(self, sessions_dir: Path, store: JsonFileStore) -> None:
        self._sessions_dir = sessions_dir
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._store = store

    def create(self, title: str | None = None) -> dict[str, Any]:
        session_id = self._new_id()
        session = {
            "id": session_id,
            "title": title or f"会话{session_id}",
            "created_at": datetime.now().isoformat(),
            "messages": [],
        }
        self._store.save(self._path(session_id), session)
        return session

    def get(self, session_id: str) -> dict[str, Any] | None:
        if not _SAFE_ID.fullmatch(session_id):
            return None
        path = self._path(session_id)
        if not path.exists():
            return None
        return self._store.load(path, {})

    def save(self, session: dict[str, Any]) -> None:
        session_id = str(session.get("id", ""))
        if not _SAFE_ID.fullmatch(session_id):
            raise ValueError("invalid session id")
        self._store.save(self._path(session_id), session)

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        for path in self._sessions_dir.glob("*.json"):
            session = self._store.load(path, {})
            if not session:
                continue
            sessions.append(
                {
                    "id": session["id"],
                    "title": session["title"],
                    "created_at": session["created_at"],
                    "message_count": len(session.get("messages", [])),
                }
            )
        return sorted(sessions, key=lambda item: item["created_at"], reverse=True)

    def delete(self, session_id: str) -> bool:
        if not _SAFE_ID.fullmatch(session_id):
            return False
        path = self._path(session_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def append_exchange(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
        sources: list[dict[str, Any]],
    ) -> None:
        if not _SAFE_ID.fullmatch(session_id):
            raise ValueError("invalid session id")
        path = self._path(session_id)
        if not path.exists():
            raise ValueError("session not found")

        def append(session: dict[str, Any]) -> None:
            session.setdefault("messages", []).extend(
                [
                    {"role": "user", "content": user_content},
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "sources": sources,
                    },
                ]
            )
            title = session.get("title", "")
            if not title or title.startswith("会话"):
                session["title"] = user_content[:30] + (
                    "..." if len(user_content) > 30 else ""
                )

        self._store.mutate(path, {}, append)

    def _new_id(self) -> str:
        while True:
            candidate = uuid.uuid4().hex[:8]
            if not self._path(candidate).exists():
                return candidate

    def _path(self, session_id: str) -> Path:
        return self._sessions_dir / f"{session_id}.json"
