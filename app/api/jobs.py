from functools import lru_cache
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field, field_validator

from app.api.dependencies import get_db_session_factory
from app.core.exceptions import ConflictError, NotFoundError
from app.jobs.repository import JobRepository

router = APIRouter(prefix="/research-jobs", tags=["research jobs"])


class CreateResearchJob(BaseModel):
    question: str = Field(min_length=1, max_length=4000)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question cannot be empty")
        return value


@lru_cache
def get_job_repository() -> JobRepository:
    return JobRepository(get_db_session_factory())


@router.post("", status_code=202)
async def create_job(
    request: CreateResearchJob, repo: Annotated[JobRepository, Depends(get_job_repository)]
) -> dict[str, Any]:
    return await repo.create(request.question)


@router.get("")
async def list_jobs(
    repo: Annotated[JobRepository, Depends(get_job_repository)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[dict[str, Any]]:
    return await repo.list_recent(limit)


@router.get("/metrics")
async def job_metrics(
    repo: Annotated[JobRepository, Depends(get_job_repository)],
) -> dict[str, Any]:
    return {"jobs_by_status": await repo.counts()}


@router.get("/{job_id}")
async def get_job(
    job_id: str, repo: Annotated[JobRepository, Depends(get_job_repository)]
) -> dict[str, Any]:
    job = await repo.get(job_id)
    if job is None:
        raise NotFoundError("research job not found", code="JOB_NOT_FOUND")
    return job


@router.post("/{job_id}/cancel")
async def cancel_job(
    job_id: str, repo: Annotated[JobRepository, Depends(get_job_repository)]
) -> dict[str, bool]:
    if not await repo.cancel(job_id):
        if await repo.get(job_id) is None:
            raise NotFoundError("research job not found", code="JOB_NOT_FOUND")
        raise ConflictError("job is already finished", code="JOB_FINISHED")
    return {"cancelled": True}
