"""Run a single durable research worker: python -m app.jobs.worker."""

import asyncio
import logging
from typing import Any

from app.agent.research import build_research_graph
from app.api.dependencies import get_db_session_factory, get_llm_client, get_retriever
from app.core.config import Settings, get_settings
from app.core.database import dispose_engine
from app.core.logging import configure_logging
from app.jobs.repository import JobRepository

logger = logging.getLogger("rag_app.jobs")


class LeaseLostError(Exception):
    pass


async def execute_claim(repo: Any, graph: Any, job: dict[str, Any], settings: Settings) -> None:
    job_id, token = job["id"], job["lease_token"]

    async def run() -> dict[str, Any]:
        state = await graph.ainvoke({"question": job["question"]})
        return {
            "report": state["report"],
            "citations": state.get("citations", []),
            "review": state.get("review", {}),
            "review_rounds": state.get("review_rounds", 0),
        }

    async def heartbeat() -> None:
        while True:
            await asyncio.sleep(settings.job_lease_seconds / 3)
            if not await repo.heartbeat(job_id, token, settings.job_lease_seconds):
                raise LeaseLostError("job cancelled or lease lost")

    execution = asyncio.create_task(run())
    renewal = asyncio.create_task(heartbeat())
    try:
        done, _ = await asyncio.wait({execution, renewal}, return_when=asyncio.FIRST_COMPLETED)
        if renewal in done:
            await renewal
        result = await execution
        if await repo.complete(job_id, token, result):
            logger.info("job_completed job_id=%s", job_id)
        else:
            logger.warning("job_commit_rejected job_id=%s", job_id)
    except LeaseLostError:
        logger.info("job_lease_lost job_id=%s", job_id)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("job_failed job_id=%s", job_id)
        await repo.fail(
            job_id, token, attempts=job["attempts"], max_attempts=settings.job_max_attempts
        )
    finally:
        for task in (execution, renewal):
            if not task.done():
                task.cancel()
        await asyncio.gather(execution, renewal, return_exceptions=True)


async def main() -> None:
    configure_logging()
    settings = get_settings()
    repo = JobRepository(get_db_session_factory())
    graph = build_research_graph(
        retriever=get_retriever(), llm_client=get_llm_client(), settings=settings
    )
    logger.info("worker_started")
    try:
        while True:
            try:
                job = await repo.claim(
                    lease_seconds=settings.job_lease_seconds, max_attempts=settings.job_max_attempts
                )
                if job:
                    await execute_claim(repo, graph, job, settings)
                else:
                    await asyncio.sleep(settings.job_poll_seconds)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("worker_poll_failed")
                await asyncio.sleep(settings.job_poll_seconds)
    finally:
        await dispose_engine()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
