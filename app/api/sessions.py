from typing import Annotated

from fastapi import APIRouter, Body, Depends, status

from app.api.dependencies import get_session_service
from app.schemas.document import DeleteResponse
from app.schemas.session import CreateSessionRequest, SessionDetail, SessionSummary
from app.services.session_service import SessionService

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("", response_model=list[SessionSummary])
def list_sessions(
    service: Annotated[SessionService, Depends(get_session_service)],
) -> list[dict]:
    return service.list()


@router.post("", response_model=SessionDetail, status_code=status.HTTP_201_CREATED)
def create_session(
    service: Annotated[SessionService, Depends(get_session_service)],
    request: Annotated[CreateSessionRequest | None, Body()] = None,
) -> dict:
    return service.create(request.title if request else None)


@router.get("/{session_id}", response_model=SessionDetail)
def get_session(
    session_id: str,
    service: Annotated[SessionService, Depends(get_session_service)],
) -> dict:
    return service.get(session_id)


@router.delete("/{session_id}", response_model=DeleteResponse)
def delete_session(
    session_id: str,
    service: Annotated[SessionService, Depends(get_session_service)],
) -> DeleteResponse:
    service.delete(session_id)
    return DeleteResponse(deleted=True)
