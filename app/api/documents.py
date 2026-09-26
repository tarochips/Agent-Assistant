from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile, status

from app.api.dependencies import get_document_service
from app.schemas.document import DeleteResponse, DocumentSummary, DocumentUploadResponse
from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=list[DocumentSummary])
def list_documents(
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> list[dict]:
    return service.list()


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    service: Annotated[DocumentService, Depends(get_document_service)],
    file: Annotated[UploadFile, File()],
) -> dict:
    content = await file.read()
    return service.upload(file.filename or "", content)


@router.delete("/{doc_id}", response_model=DeleteResponse)
def delete_document(
    doc_id: str,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> DeleteResponse:
    service.delete(doc_id)
    return DeleteResponse(deleted=True)
