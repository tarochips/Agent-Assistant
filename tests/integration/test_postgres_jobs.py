"""Opt-in PostgreSQL queue test; use a dedicated TEST_DATABASE_URL database."""

import asyncio
import os
import uuid
from typing import cast

import pytest
from sqlalchemy import Table, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.models import ResearchJobRecord
from app.jobs.repository import JobRepository


@pytest.mark.asyncio
async def test_postgres_claim_fencing_and_cancel():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set TEST_DATABASE_URL to a dedicated PostgreSQL test database")
    schema = "rag_test_" + uuid.uuid4().hex
    admin = create_async_engine(url)
    async with admin.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_async_engine(url, connect_args={"server_settings": {"search_path": schema}})
    try:
        async with engine.begin() as connection:
            await connection.run_sync(cast(Table, ResearchJobRecord.__table__).create)
        repo = JobRepository(async_sessionmaker(engine, expire_on_commit=False))
        created = await repo.create("test")
        claims = await asyncio.gather(
            *[repo.claim(lease_seconds=1, max_attempts=3) for _ in range(2)]
        )
        first = next(job for job in claims if job)
        assert sum(job is not None for job in claims) == 1
        await asyncio.sleep(1.2)
        second = await repo.claim(lease_seconds=10, max_attempts=3)
        assert second is not None and second["lease_token"] != first["lease_token"]
        assert not await repo.complete(created["id"], first["lease_token"], {"report": "old"})
        assert await repo.cancel(created["id"])
        assert not await repo.complete(created["id"], second["lease_token"], {"report": "new"})
        saved = await repo.get(created["id"])
        assert saved is not None and saved["status"] == "cancelled"
    finally:
        await engine.dispose()
        async with admin.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()


@pytest.mark.asyncio
async def test_postgres_summary_compare_and_swap():
    from app.db.models import MessageRecord, SessionRecord
    from app.repositories.postgres import PostgresSessionRepository

    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set TEST_DATABASE_URL to a dedicated PostgreSQL test database")
    schema = "rag_test_" + uuid.uuid4().hex
    admin = create_async_engine(url)
    async with admin.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_async_engine(url, connect_args={"server_settings": {"search_path": schema}})
    try:
        async with engine.begin() as connection:
            await connection.run_sync(cast(Table, SessionRecord.__table__).create)
            await connection.run_sync(cast(Table, MessageRecord.__table__).create)
        repo = PostgresSessionRepository(async_sessionmaker(engine, expire_on_commit=False))
        session = await repo.create()
        await repo.append_exchange(session["id"], "问题", "回答", [])
        results = await asyncio.gather(
            *[
                repo.update_summary(session["id"], expected_count=0, new_count=2, summary=summary)
                for summary in ("summary A", "summary B")
            ]
        )
        assert sorted(results) == [False, True]
        saved = await repo.get(session["id"])
        assert saved is not None and saved["summary_message_count"] == 2
        assert len(saved["messages"]) == 2
    finally:
        await engine.dispose()
        async with admin.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()
