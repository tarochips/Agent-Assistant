from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_chat_service
from app.schemas.chat import ChatRequest, PendingApprovalResponse, ResumeChatRequest
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/stream")
async def stream_chat(
    request: ChatRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> StreamingResponse:
    return StreamingResponse(
        service.stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/{session_id}/resume")
async def resume_chat(
    session_id: str,
    request: ResumeChatRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> StreamingResponse:
    await service.ensure_resumable(session_id)
    return StreamingResponse(
        service.resume(session_id, request.approved),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{session_id}/pending", response_model=PendingApprovalResponse)
async def get_pending_approval(
    session_id: str,
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> PendingApprovalResponse:
    return PendingApprovalResponse(pending=await service.pending_approval(session_id))
