from functools import lru_cache

from app.core.config import get_settings
from app.llm.client import DeepSeekClient
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService
from app.services.session_service import SessionService


@lru_cache
def get_db_session_factory():
    from app.core.database import get_session_factory

    return get_session_factory()


@lru_cache
def get_document_repository():
    from app.repositories.postgres import PostgresDocumentRepository

    return PostgresDocumentRepository(get_db_session_factory())


@lru_cache
def get_session_repository():
    from app.repositories.postgres import PostgresSessionRepository

    return PostgresSessionRepository(get_db_session_factory())


@lru_cache
def get_retriever():
    from app.retrieval.vector import VectorRetriever
    from app.retrieval.vectorizer import LocalHashVectorizer

    settings = get_settings()
    return VectorRetriever(
        get_document_repository(),
        LocalHashVectorizer(),
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
