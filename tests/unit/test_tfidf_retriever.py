from pathlib import Path

from app.repositories.documents import JsonDocumentRepository
from app.repositories.json_io import JsonFileStore
from app.retrieval.tfidf import TfidfRetriever


def build_retriever(tmp_path: Path) -> tuple[TfidfRetriever, JsonDocumentRepository, Path]:
    store = JsonFileStore()
    repository = JsonDocumentRepository(tmp_path / "docstore.json", store)
    vectorizer_path = tmp_path / "vectorizer.pkl"
    retriever = TfidfRetriever(
        repository,
        vectorizer_path,
        chunk_size=100,
        chunk_overlap=20,
    )
    return retriever, repository, vectorizer_path


def test_retriever_ranks_related_document_first(tmp_path: Path) -> None:
    retriever, _repository, _path = build_retriever(tmp_path)
    fruit_id, _ = retriever.index_document("fruit.md", "苹果和香蕉都是常见水果。")
    retriever.index_document("database.md", "数据库用于持久化结构化业务数据。")

    results = retriever.search("苹果水果", top_k=2, min_score=0.0)

    assert results[0].document_id == fruit_id
    assert results[0].filename == "fruit.md"
    assert results[0].score >= results[1].score


def test_retriever_applies_threshold(tmp_path: Path) -> None:
    retriever, _repository, _path = build_retriever(tmp_path)
    retriever.index_document("fruit.md", "苹果和香蕉都是常见水果。")

    assert retriever.search("完全无关的问题", top_k=4, min_score=0.5) == []


def test_deleting_last_document_clears_index(tmp_path: Path) -> None:
    retriever, repository, vectorizer_path = build_retriever(tmp_path)
    doc_id, _ = retriever.index_document("fruit.md", "苹果和香蕉都是常见水果。")
    assert vectorizer_path.exists()

    assert retriever.remove_document(doc_id) is True
    assert repository.get_all_chunks() == []
    assert not vectorizer_path.exists()
