import numpy as np
import pytest

from app.retrieval.models import SearchResult
from app.retrieval.vector import VectorRetriever
from app.retrieval.vectorizer import LocalHashVectorizer


class MemoryVectorRepository:
    def __init__(self) -> None:
        self.documents: dict[str, list[tuple[str, list[float]]]] = {}

    async def add_document(self, filename, chunks, vectors):
        doc_id = str(len(self.documents) + 1)
        self.documents[doc_id] = list(zip(chunks, vectors, strict=True))
        return doc_id

    async def remove_document(self, doc_id):
        return self.documents.pop(doc_id, None) is not None

    async def has_chunks(self):
        return bool(self.documents)

    async def search(self, query_vector, *, top_k, min_score):
        query = np.asarray(query_vector)
        results = []
        for doc_id, chunks in self.documents.items():
            for chunk_index, (text, vector) in enumerate(chunks):
                candidate = np.asarray(vector)
                score = float(np.dot(query, candidate))
                if score >= min_score:
                    results.append(
                        SearchResult(
                            chunk_id=f"{doc_id}:{chunk_index}",
                            document_id=doc_id,
                            filename=f"{doc_id}.md",
                            chunk_index=chunk_index,
                            text=text,
                            score=score,
                        )
                    )
        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]


async def test_local_vectorizer_normalizes_vectors() -> None:
    vectorizer = LocalHashVectorizer()

    vectors = await vectorizer.embed(["PostgreSQL stores structured data.", ""])

    assert len(vectors) == 2
    assert len(vectors[0]) == LocalHashVectorizer.DIMENSIONS
    assert np.linalg.norm(vectors[0]) == pytest.approx(1.0)
    assert np.linalg.norm(vectors[1]) == 0.0


async def test_vector_retriever_indexes_and_ranks_local_vectors() -> None:
    repository = MemoryVectorRepository()
    retriever = VectorRetriever(
        repository,
        LocalHashVectorizer(),
        chunk_size=200,
        chunk_overlap=20,
    )
    database_id, _ = await retriever.index_document(
        "database.md", "PostgreSQL stores structured customer order data."
    )
    await retriever.index_document("garden.md", "Rabbits eat carrots in the garden.")

    results = await retriever.search("customer database orders", top_k=2, min_score=0.0)

    assert results[0].document_id == database_id
    assert await retriever.remove_document(database_id) is True
