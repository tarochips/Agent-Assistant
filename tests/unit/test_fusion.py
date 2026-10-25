import pytest

from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.models import SearchResult


def result(chunk_id: str, text: str) -> SearchResult:
    document_id, chunk_index = chunk_id.split(":")
    return SearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        filename=f"{document_id}.md",
        chunk_index=int(chunk_index),
        text=text,
        score=1.0,
    )


def test_rrf_promotes_chunks_ranked_by_both_retrievers() -> None:
    dense = [result("1:0", "shared"), result("2:0", "second")]
    lexical = [result("2:0", "shared"), result("1:0", "second")]

    fused = reciprocal_rank_fusion([dense, lexical], rrf_k=1, limit=5)

    assert [item.chunk_id for item in fused] == ["1:0", "2:0"]
    assert all(0.0 < item.score <= 1.0 for item in fused)
    assert fused[0].score == fused[1].score


def test_rrf_handles_empty_rankings_and_validates_parameters() -> None:
    assert reciprocal_rank_fusion([[], []], rrf_k=60, limit=5) == []
    with pytest.raises(ValueError, match="rrf_k"):
        reciprocal_rank_fusion([], rrf_k=0, limit=5)
