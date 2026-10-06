from typing import Any

from app.retrieval.models import SearchResult
from app.retrieval.tfidf import split_text
from app.retrieval.vectorizer import TextVectorizer


class VectorRetriever:
    def __init__(
        self,
        repository: Any,
        embedding_client: TextVectorizer,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self._repository = repository
        self._embedding_client = embedding_client
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    async def index_document(self, filename: str, content: str) -> tuple[str, int]:
        chunks = split_text(
            content,
            chunk_size=self._chunk_size,
            overlap=self._chunk_overlap,
        )
        if not chunks:
            raise ValueError("document contains no indexable text")
        vectors = await self._embedding_client.embed(chunks)
        doc_id = await self._repository.add_document(filename, chunks, vectors)
        return doc_id, len(chunks)

    async def remove_document(self, doc_id: str) -> bool:
        return await self._repository.remove_document(doc_id)

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        min_score: float,
    ) -> list[SearchResult]:
        if not await self._repository.has_chunks():
            return []
        query_vector = (await self._embedding_client.embed([query]))[0]
        return await self._repository.search(
            query_vector,
            top_k=top_k,
            min_score=min_score,
        )
