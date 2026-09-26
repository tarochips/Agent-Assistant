from functools import lru_cache

from app.core.config import get_settings
from app.llm.client import DeepSeekClient
from app.repositories.documents import JsonDocumentRepository
from app.repositories.json_io import JsonFileStore
from app.repositories.sessions import JsonSessionRepository
from app.retrieval.tfidf import TfidfRetriever
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService
from app.services.session_service import SessionService


@lru_cache
def get_json_store() -> JsonFileStore:
    return JsonFileStore()


@lru_cache
def get_document_repository() -> JsonDocumentRepository:
    settings = get_settings()
    return JsonDocumentRepository(settings.docstore_path, get_json_store())


@lru_cache
def get_session_repository() -> JsonSessionRepository:
    settings = get_settings()
    return JsonSessionRepository(settings.sessions_dir, get_json_store())


@lru_cache
def get_retriever() -> TfidfRetriever:
    settings = get_settings()
    return TfidfRetriever(
        get_document_repository(),
        settings.vectorizer_path,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


@lru_cache
def get_llm_client() -> DeepSeekClient:
    return DeepSeekClient(get_settings())


@lru_cache
def get_chat_service() -> ChatService:
    return ChatService(
        get_session_repository(),
        get_retriever(),
        get_llm_client(),
        get_settings(),
    )


@lru_cache
def get_document_service() -> DocumentService:
    return DocumentService(
        get_document_repository(),
        get_retriever(),
        get_settings(),
    )


@lru_cache
def get_session_service() -> SessionService:
    return SessionService(get_session_repository())
