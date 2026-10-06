from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.exceptions import InvalidDocumentError, NotFoundError
from app.services._async_utils import resolve


class DocumentService:
    ALLOWED_SUFFIXES = {".txt", ".md"}

    def __init__(
        self,
        repository: Any,
        retriever: Any,
        settings: Settings,
    ) -> None:
        self._repository = repository
        self._retriever = retriever
        self._settings = settings

    async def list(self) -> list[dict[str, Any]]:
        return await resolve(self._repository.list_documents())

    async def upload(self, filename: str, raw_content: bytes) -> dict[str, Any]:
        safe_filename = Path(filename).name.strip()
        if not safe_filename:
            raise InvalidDocumentError("filename is required")
        if Path(safe_filename).suffix.lower() not in self.ALLOWED_SUFFIXES:
            raise InvalidDocumentError("V1 only supports UTF-8 .txt and .md files")
        if not raw_content:
            raise InvalidDocumentError("document is empty")
        if len(raw_content) > self._settings.max_upload_bytes:
            raise InvalidDocumentError(
                f"document exceeds {self._settings.max_upload_bytes} bytes"
            )

        try:
            content = raw_content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise InvalidDocumentError("document must be UTF-8 encoded") from exc
        if not content.strip():
            raise InvalidDocumentError("document contains no text")

        doc_id, chunk_count = await resolve(
            self._retriever.index_document(safe_filename, content)
        )
        return {
            "doc_id": doc_id,
            "filename": safe_filename,
            "chunk_count": chunk_count,
        }

    async def delete(self, doc_id: str) -> None:
        if not await resolve(self._retriever.remove_document(doc_id)):
            raise NotFoundError("document not found", code="DOCUMENT_NOT_FOUND")
