"""PostgreSQL durable job storage with leased, fenced claims."""

import uuid
from datetime import timedelta
from typing import Any

from sqlalchemy import Interval, and_, func, literal, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import ResearchJobRecord


class JobRepository:
    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        self.factory = factory

    async def create(self, question: str) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        async with self.factory() as session, session.begin():
            job = ResearchJobRecord(
                id=job_id,
                question=question,
                status="queued",
                attempts=0,
                created_at=func.now(),
                updated_at=func.now(),
            )
            session.add(job)
            await session.flush()
            await session.refresh(job)
            return self.serialize(job)

    async def get(self, job_id: str) -> dict[str, Any] | None:
        async with self.factory() as session:
            job = await session.get(ResearchJobRecord, job_id)
            return self.serialize(job) if job else None

    async def list_recent(self, limit: int) -> list[dict[str, Any]]:
        async with self.factory() as session:
            result = await session.execute(
                select(ResearchJobRecord)
                .order_by(ResearchJobRecord.created_at.desc(), ResearchJobRecord.id.desc())
                .limit(limit)
            )
            return [self.serialize(job) for job in result.scalars().all()]

    async def claim(self, *, lease_seconds: int, max_attempts: int) -> dict[str, Any] | None:
        async with self.factory() as session, session.begin():
            # Exhausted leases cannot be picked up again, including after a worker crash.
            await session.execute(
                update(ResearchJobRecord)
                .where(
                    ResearchJobRecord.status == "running",
                    ResearchJobRecord.lease_until <= func.now(),
                    ResearchJobRecord.attempts >= max_attempts,
                )
                .values(
                    status="failed",
                    error="任务执行次数已耗尽。",
                    lease_token=None,
                    lease_until=None,
                    updated_at=func.now(),
                )
            )
            statement = (
                select(ResearchJobRecord)
                .where(
                    ResearchJobRecord.attempts < max_attempts,
                    or_(
                        ResearchJobRecord.status == "queued",
                        and_(
                            ResearchJobRecord.status == "running",
                            ResearchJobRecord.lease_until <= func.now(),
                        ),
                    ),
                )
                .order_by(ResearchJobRecord.created_at, ResearchJobRecord.id)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            job = (await session.execute(statement)).scalar_one_or_none()
            if job is None:
                return None
            token = uuid.uuid4().hex
            job.status = "running"
            job.attempts += 1
            job.error = None
            job.lease_token = token
            job.lease_until = func.now() + literal(
                timedelta(seconds=lease_seconds), type_=Interval()
            )
            job.updated_at = func.now()
            await session.flush()
            return {
                "id": job.id,
                "question": job.question,
                "lease_token": token,
                "attempts": job.attempts,
            }

    async def heartbeat(self, job_id: str, token: str, lease_seconds: int) -> bool:
        return await self._owned_update(
            job_id,
            token,
            {
                "lease_until": func.now()
                + literal(timedelta(seconds=lease_seconds), type_=Interval()),
                "updated_at": func.now(),
            },
        )

    async def complete(self, job_id: str, token: str, result: dict[str, Any]) -> bool:
        return await self._owned_update(
            job_id,
            token,
            {
                "status": "completed",
                "result": result,
                "error": None,
                "lease_token": None,
                "lease_until": None,
                "updated_at": func.now(),
            },
        )

    async def fail(self, job_id: str, token: str, *, attempts: int, max_attempts: int) -> bool:
        return await self._owned_update(
            job_id,
            token,
            {
                "status": "queued" if attempts < max_attempts else "failed",
                "error": "任务执行失败，请检查服务日志。",
                "lease_token": None,
                "lease_until": None,
                "updated_at": func.now(),
            },
        )

    async def _owned_update(self, job_id: str, token: str, values: dict[str, Any]) -> bool:
        async with self.factory() as session, session.begin():
            result = await session.execute(
                update(ResearchJobRecord)
                .where(
                    ResearchJobRecord.id == job_id,
                    ResearchJobRecord.status == "running",
                    ResearchJobRecord.lease_token == token,
                    ResearchJobRecord.lease_until > func.now(),
                )
                .values(**values)
                .returning(ResearchJobRecord.id)
            )
            return result.scalar_one_or_none() is not None

    async def cancel(self, job_id: str) -> bool:
        async with self.factory() as session, session.begin():
            result = await session.execute(
                update(ResearchJobRecord)
                .where(
                    ResearchJobRecord.id == job_id,
                    ResearchJobRecord.status.in_(["queued", "running"]),
                )
                .values(
                    status="cancelled", lease_token=None, lease_until=None, updated_at=func.now()
                )
                .returning(ResearchJobRecord.id)
            )
            return result.scalar_one_or_none() is not None

    async def counts(self) -> dict[str, int]:
        async with self.factory() as session:
            result = await session.execute(
                select(ResearchJobRecord.status, func.count()).group_by(ResearchJobRecord.status)
            )
            return {status: count for status, count in result.all()}

    @staticmethod
    def serialize(job: ResearchJobRecord) -> dict[str, Any]:
        return {
            "id": job.id,
            "question": job.question,
            "status": job.status,
            "attempts": job.attempts,
            "result": job.result,
            "error": job.error,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }
