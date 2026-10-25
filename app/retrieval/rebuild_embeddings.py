"""Re-embed existing PostgreSQL chunks after changing the embedding model."""

import asyncio

from app.core.config import get_settings
from app.core.database import dispose_engine, get_session_factory
from app.repositories.postgres import PostgresDocumentRepository
from app.retrieval.vector import VectorRetriever
from app.retrieval.vectorizer import FastEmbedE5Vectorizer


async def rebuild_embeddings() -> int:
    settings = get_settings()
    retriever = VectorRetriever(
        PostgresDocumentRepository(get_session_factory()),
        FastEmbedE5Vectorizer(
            settings.embedding_cache_dir,
            threads=settings.embedding_threads,
            model_name=settings.embedding_model,
        ),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        candidate_k=settings.retrieval_candidate_k,
        rrf_k=settings.retrieval_rrf_k,
    )
    try:
        return await retriever.rebuild_embeddings()
    finally:
        await dispose_engine()


async def main() -> None:
    updated_count = await rebuild_embeddings()
    print(f"Embedding rebuild complete: {updated_count} chunks updated.")


if __name__ == "__main__":
    asyncio.run(main())
