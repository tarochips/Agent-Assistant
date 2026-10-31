from functools import lru_cache

from fastapi import Request

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
    from app.retrieval.reranker import FastEmbedCrossEncoderReranker, NoOpReranker
    from app.retrieval.vector import VectorRetriever
    from app.retrieval.vectorizer import FastEmbedE5Vectorizer

    settings = get_settings()
    reranker = (
        FastEmbedCrossEncoderReranker(
            str(settings.embedding_cache_dir),
            model_name=settings.retrieval_reranker_model,
            threads=settings.embedding_threads,
        )
        if settings.retrieval_reranker_enabled
        else NoOpReranker()
    )
    return VectorRetriever(
        get_document_repository(),
        FastEmbedE5Vectorizer(
            settings.embedding_cache_dir,
            threads=settings.embedding_threads,
            model_name=settings.embedding_model,
        ),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        candidate_k=settings.retrieval_candidate_k,
        rrf_k=settings.retrieval_rrf_k,
        reranker=reranker,
    )


@lru_cache
def get_llm_client() -> DeepSeekClient:
    return DeepSeekClient(get_settings())


def get_chat_service(request: Request) -> ChatService:
    return ChatService(
        get_session_repository(),
        request.app.state.agent_graph,
        get_settings(),
        request.app.state.research_graph,
        get_llm_client(),
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
