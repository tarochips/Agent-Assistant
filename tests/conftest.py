from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_chat_service,
    get_document_service,
    get_session_service,
)
from app.core.config import PROJECT_ROOT, Settings
from app.main import create_app
from app.repositories.documents import JsonDocumentRepository
from app.repositories.json_io import JsonFileStore
from app.repositories.sessions import JsonSessionRepository
from app.retrieval.tfidf import TfidfRetriever
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService
from app.services.session_service import SessionService


class FakeLLMClient:
    async def stream(self, _messages: list[dict[str, str]]) -> AsyncIterator[str]:
        yield "测试"
        yield "回答"


@pytest.fixture
def service_bundle(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        static_dir=PROJECT_ROOT / "static",
        deepseek_api_key="test-key",
        chunk_size=40,
        chunk_overlap=10,
        retrieval_top_k=3,
        retrieval_min_score=0.01,
    )
    store = JsonFileStore()
    document_repository = JsonDocumentRepository(settings.docstore_path, store)
    session_repository = JsonSessionRepository(settings.sessions_dir, store)
    retriever = TfidfRetriever(
        document_repository,
        settings.vectorizer_path,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return {
        "settings": settings,
        "document_repository": document_repository,
        "session_repository": session_repository,
        "retriever": retriever,
        "document_service": DocumentService(document_repository, retriever, settings),
        "session_service": SessionService(session_repository),
        "chat_service": ChatService(
            session_repository,
            retriever,
            FakeLLMClient(),
            settings,
        ),
    }


@pytest.fixture
def client(service_bundle):
    application = create_app()
    application.dependency_overrides[get_document_service] = (
        lambda: service_bundle["document_service"]
    )
    application.dependency_overrides[get_session_service] = (
        lambda: service_bundle["session_service"]
    )
    application.dependency_overrides[get_chat_service] = lambda: service_bundle["chat_service"]
    with TestClient(application) as test_client:
        yield test_client
    application.dependency_overrides.clear()
