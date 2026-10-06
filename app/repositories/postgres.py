import re
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import ChunkRecord, DocumentRecord, MessageRecord, SessionRecord
from app.retrieval.models import SearchResult

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _as_datetime(value: str | datetime | None) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        result = datetime.fromisoformat(value)
    else:
        result = datetime.now(UTC)
    if result.tzinfo is None:
        result = result.replace(tzinfo=UTC)
    return result


class PostgresDocumentRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def list_documents(self) -> list[dict[str, Any]]:
        chunk_count = func.count(ChunkRecord.id).label("chunk_count")
        statement = (
            select(DocumentRecord, chunk_count)
            .outerjoin(ChunkRecord, ChunkRecord.document_id == DocumentRecord.id)
            .group_by(DocumentRecord.id)
            .order_by(DocumentRecord.uploaded_at.desc())
        )
        async with self._session_factory() as session:
            rows = (await session.execute(statement)).all()
        return [
            {
                "doc_id": document.id,
                "filename": document.filename,
                "chunk_count": int(count),
                "uploaded_at": document.uploaded_at.isoformat(),
            }
            for document, count in rows
        ]

    async def add_document(
        self,
        filename: str,
        chunk_texts: list[str],
        embeddings: list[list[float]],
        *,
        doc_id: str | None = None,
        uploaded_at: str | datetime | None = None,
    ) -> str:
        if len(chunk_texts) != len(embeddings):
            raise ValueError("each document chunk must have one embedding")
        if not chunk_texts:
            raise ValueError("document must contain at least one chunk")

        actual_id = doc_id or uuid.uuid4().hex[:8]
        if not _SAFE_ID.fullmatch(actual_id):
            raise ValueError("invalid document id")
        document = DocumentRecord(
            id=actual_id,
            filename=filename,
            uploaded_at=_as_datetime(uploaded_at),
        )
        chunks = [
            ChunkRecord(
                document_id=actual_id,
                chunk_index=index,
                text=chunk_text,
                embedding=vector,
            )
            for index, (chunk_text, vector) in enumerate(zip(chunk_texts, embeddings, strict=True))
        ]
        async with self._session_factory() as session, session.begin():
            session.add(document)
            await session.flush()
            session.add_all(chunks)
        return actual_id

    async def document_exists(self, doc_id: str) -> bool:
        if not _SAFE_ID.fullmatch(doc_id):
            return False
        async with self._session_factory() as session:
            return await session.get(DocumentRecord, doc_id) is not None

    async def remove_document(self, doc_id: str) -> bool:
        if not _SAFE_ID.fullmatch(doc_id):
            return False
        async with self._session_factory() as session, session.begin():
            document = await session.get(DocumentRecord, doc_id)
            if document is None:
                return False
            await session.delete(document)
        return True

    async def has_chunks(self) -> bool:
        async with self._session_factory() as session:
            result = await session.execute(select(ChunkRecord.id).limit(1))
            return result.scalar_one_or_none() is not None

    async def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        min_score: float,
    ) -> list[SearchResult]:
        distance = ChunkRecord.embedding.cosine_distance(query_embedding)
        statement = (
            select(ChunkRecord, DocumentRecord.filename, distance.label("distance"))
            .join(DocumentRecord, DocumentRecord.id == ChunkRecord.document_id)
            .where(distance <= 1.0 - min_score)
            .order_by(distance.asc())
            .limit(top_k)
        )
        async with self._session_factory() as session:
            rows = (await session.execute(statement)).all()

        results: list[SearchResult] = []
        for chunk, filename, raw_distance in rows:
            score = max(0.0, min(1.0, 1.0 - float(raw_distance)))
            results.append(
                SearchResult(
                    chunk_id=f"{chunk.document_id}:{chunk.chunk_index}",
                    document_id=chunk.document_id,
                    filename=filename,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    score=score,
                )
            )
        return results


class PostgresSessionRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def create(self, title: str | None = None) -> dict[str, Any]:
        session_id = self._new_id()
        record = SessionRecord(
            id=session_id,
            title=title or f"会话{session_id}",
            created_at=datetime.now(UTC),
        )
        async with self._session_factory() as session, session.begin():
            session.add(record)
        return await self.get(session_id) or {}

    async def get(self, session_id: str) -> dict[str, Any] | None:
        if not _SAFE_ID.fullmatch(session_id):
            return None
        async with self._session_factory() as session:
            record = await session.get(SessionRecord, session_id)
            if record is None:
                return None
            statement = (
                select(MessageRecord)
                .where(MessageRecord.session_id == session_id)
                .order_by(MessageRecord.id.asc())
            )
            messages = (await session.execute(statement)).scalars().all()
            return {
                "id": record.id,
                "title": record.title,
                "created_at": record.created_at.isoformat(),
                "messages": [
                    {
                        "role": message.role,
                        "content": message.content,
                        "sources": message.sources or [],
                    }
                    for message in messages
                ],
            }

    async def list_sessions(self) -> list[dict[str, Any]]:
        count = func.count(MessageRecord.id).label("message_count")
        statement = (
            select(SessionRecord, count)
            .outerjoin(MessageRecord, MessageRecord.session_id == SessionRecord.id)
            .group_by(SessionRecord.id)
            .order_by(SessionRecord.created_at.desc())
        )
        async with self._session_factory() as session:
            rows = (await session.execute(statement)).all()
        return [
            {
                "id": record.id,
                "title": record.title,
                "created_at": record.created_at.isoformat(),
                "message_count": int(message_count),
            }
            for record, message_count in rows
        ]

    async def delete(self, session_id: str) -> bool:
        if not _SAFE_ID.fullmatch(session_id):
            return False
        async with self._session_factory() as session, session.begin():
            record = await session.get(SessionRecord, session_id)
            if record is None:
                return False
            await session.delete(record)
        return True

    async def append_exchange(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
        sources: list[dict[str, Any]],
    ) -> None:
        if not _SAFE_ID.fullmatch(session_id):
            raise ValueError("invalid session id")
        async with self._session_factory() as session, session.begin():
            record = await session.get(SessionRecord, session_id)
            if record is None:
                raise ValueError("session not found")
            session.add_all(
                [
                    MessageRecord(session_id=session_id, role="user", content=user_content),
                    MessageRecord(
                        session_id=session_id,
                        role="assistant",
                        content=assistant_content,
                        sources=sources,
                    ),
                ]
            )
            if not record.title or record.title.startswith("会话"):
                record.title = user_content[:30] + ("..." if len(user_content) > 30 else "")

    async def import_session(self, data: dict[str, Any]) -> bool:
        session_id = str(data.get("id", ""))
        if not _SAFE_ID.fullmatch(session_id):
            return False
        async with self._session_factory() as session, session.begin():
            if await session.get(SessionRecord, session_id) is not None:
                return False
            record = SessionRecord(
                id=session_id,
                title=str(data.get("title") or f"会话{session_id}"),
                created_at=_as_datetime(data.get("created_at")),
            )
            session.add(record)
            for message in data.get("messages", []):
                role = message.get("role")
                content = message.get("content")
                if role not in {"user", "assistant"} or not isinstance(content, str):
                    continue
                session.add(
                    MessageRecord(
                        session_id=session_id,
                        role=role,
                        content=content,
                        sources=message.get("sources", []) if role == "assistant" else [],
                    )
                )
        return True

    @staticmethod
    def _new_id() -> str:
        return uuid.uuid4().hex[:8]
