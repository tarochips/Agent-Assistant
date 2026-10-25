import pytest

from app.retrieval.models import SearchResult
from app.retrieval.reranker import FastEmbedCrossEncoderReranker


def make_result(chunk_id: str, text: str) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id="doc",
        filename="doc.md",
        chunk_index=0,
        text=text,
        score=0.5,
    )


async def test_cross_encoder_reranks_candidates_and_normalizes_scores() -> None:
    class FakeModel:
        def rerank(self, query, documents):
            assert query == "database"
            assert documents == ["unrelated", "database guide"]
            return [-2.0, 2.0]

    reranker = FastEmbedCrossEncoderReranker("cache")
    reranker._model = FakeModel()

    results = await reranker.rerank(
        "database",
        [make_result("doc:0", "unrelated"), make_result("doc:1", "database guide")],
    )

    assert [result.chunk_id for result in results] == ["doc:1", "doc:0"]
    assert results[0].score == pytest.approx(0.8808, abs=1e-4)
    assert results[1].score == pytest.approx(0.1192, abs=1e-4)


async def test_cross_encoder_rejects_score_count_mismatch() -> None:
    class FakeModel:
        def rerank(self, _query, _documents):
            return []

    reranker = FastEmbedCrossEncoderReranker("cache")
    reranker._model = FakeModel()

    with pytest.raises(ValueError, match="different number of scores"):
        await reranker.rerank("query", [make_result("doc:0", "text")])
