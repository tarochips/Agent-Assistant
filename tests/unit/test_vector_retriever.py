import numpy as np
import pytest

from app.retrieval.models import SearchResult
from app.retrieval.reranker import TextReranker
from app.retrieval.vector import StaleEmbeddingError, VectorRetriever
from app.retrieval.vectorizer import FastEmbedE5Vectorizer, LocalHashVectorizer


class MemoryVectorRepository:
    def __init__(self) -> None:
        self.chunks: dict[str, tuple[SearchResult, list[float], str]] = {}

    async def add_document(self, filename, chunks, vectors, *, embedding_model):
        doc_id = str(len(self.chunks) + 1)
        for chunk_index, (text, vector) in enumerate(zip(chunks, vectors, strict=True)):
            result = SearchResult(
                chunk_id=f"{doc_id}:{chunk_index}",
                document_id=doc_id,
                filename=filename,
                chunk_index=chunk_index,
                text=text,
                score=0.0,
            )
            self.chunks[result.chunk_id] = (result, vector, embedding_model)
        return doc_id

    async def remove_document(self, doc_id):
        matching_ids = [
            chunk_id
            for chunk_id, (chunk, _vector, _model) in self.chunks.items()
            if chunk.document_id == doc_id
        ]
        for chunk_id in matching_ids:
            del self.chunks[chunk_id]
        return bool(matching_ids)

    async def list_chunks(self):
        return [chunk for chunk, _vector, _model in self.chunks.values()]

    async def has_stale_embeddings(self, embedding_model):
        return any(model != embedding_model for _chunk, _vector, model in self.chunks.values())

    async def search(self, query_vector, *, top_k, min_score):
        query = np.asarray(query_vector)
        ranked: list[SearchResult] = []
        for chunk, vector, _model in self.chunks.values():
            score = float(np.dot(query, np.asarray(vector)))
            if score >= min_score:
                ranked.append(
                    SearchResult(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        filename=chunk.filename,
                        chunk_index=chunk.chunk_index,
                        text=chunk.text,
                        score=score,
                    )
                )
        return sorted(ranked, key=lambda result: result.score, reverse=True)[:top_k]

    async def update_embeddings(self, embeddings, *, embedding_model):
        for chunk_id, vector in embeddings.items():
            chunk, _old_vector, _old_model = self.chunks[chunk_id]
            self.chunks[chunk_id] = (chunk, vector, embedding_model)


class FixedReranker(TextReranker):
    model_id = "test-reranker"

    async def rerank(self, _query, candidates):
        return list(reversed(candidates))


async def test_local_vectorizer_normalizes_vectors() -> None:
    vectorizer = LocalHashVectorizer()

    vectors = await vectorizer.embed(["PostgreSQL stores structured data.", ""])

    assert len(vectors) == 2
    assert len(vectors[0]) == LocalHashVectorizer.DIMENSIONS
    assert np.linalg.norm(vectors[0]) == pytest.approx(1.0)
    assert np.linalg.norm(vectors[1]) == 0.0


async def test_e5_vectorizer_uses_query_and_passage_prefixes() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.inputs: list[str] = []

        def embed(self, texts):
            self.inputs.extend(texts)
            return [np.ones(FastEmbedE5Vectorizer.DIMENSIONS, dtype=np.float32) for _ in texts]

    from pathlib import Path

    vectorizer = FastEmbedE5Vectorizer(Path("cache"))
    model = FakeModel()
    vectorizer._model = model

    await vectorizer.embed(["中文检索"], input_kind="query")
    await vectorizer.embed(["知识库内容"], input_kind="passage")

    assert model.inputs == ["query: 中文检索", "passage: 知识库内容"]
    assert vectorizer.model_id.endswith(":v1")


async def test_vector_retriever_indexes_hybrid_search_and_reranks() -> None:
    repository = MemoryVectorRepository()
    retriever = VectorRetriever(
        repository,
        LocalHashVectorizer(),
        chunk_size=200,
        chunk_overlap=20,
        candidate_k=10,
        reranker=FixedReranker(),
    )
    database_id, _ = await retriever.index_document(
        "database.md", "PostgreSQL stores structured customer order data."
    )
    await retriever.index_document("garden.md", "Rabbits eat carrots in the garden.")

    results = await retriever.search("customer database orders", top_k=2, min_score=0.0)

    assert results[0].filename == "garden.md"  # The fake reranker reverses the fused order.
    assert any(result.document_id == database_id for result in results)
    assert await retriever.remove_document(database_id) is True


async def test_stale_vectors_require_rebuild_before_search() -> None:
    repository = MemoryVectorRepository()
    vectorizer = LocalHashVectorizer()
    await repository.add_document(
        "legacy.md",
        ["Existing knowledge chunk"],
        await vectorizer.embed(["Existing knowledge chunk"]),
        embedding_model="legacy-char-hash-384-v1",
    )
    retriever = VectorRetriever(
        repository,
        vectorizer,
        chunk_size=100,
        chunk_overlap=10,
    )

    with pytest.raises(StaleEmbeddingError, match="rebuild_embeddings"):
        await retriever.search("knowledge", top_k=2, min_score=0.0)


async def test_rebuild_embeddings_updates_models_in_batches() -> None:
    repository = MemoryVectorRepository()
    vectorizer = LocalHashVectorizer()
    await repository.add_document(
        "legacy.md",
        ["first chunk", "second chunk"],
        await vectorizer.embed(["first chunk", "second chunk"]),
        embedding_model="legacy-char-hash-384-v1",
    )
    retriever = VectorRetriever(
        repository,
        vectorizer,
        chunk_size=100,
        chunk_overlap=10,
    )

    updated = await retriever.rebuild_embeddings(batch_size=1)

    assert updated == 2
    assert await repository.has_stale_embeddings(vectorizer.model_id) is False
