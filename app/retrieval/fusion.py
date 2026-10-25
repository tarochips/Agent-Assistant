"""Rank fusion helpers for combining independent retrieval methods."""

from collections.abc import Sequence
from dataclasses import replace

from app.retrieval.models import SearchResult


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchResult]],
    *,
    rrf_k: int = 60,
    limit: int,
) -> list[SearchResult]:
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    if limit <= 0:
        return []

    active_rankings = [ranking for ranking in rankings if ranking]
    if not active_rankings:
        return []

    fused: dict[str, tuple[SearchResult, float]] = {}
    for ranking in active_rankings:
        for rank, result in enumerate(ranking, start=1):
            existing, score = fused.get(result.chunk_id, (result, 0.0))
            fused[result.chunk_id] = (existing, score + 1.0 / (rrf_k + rank))

    maximum_score = len(active_rankings) / (rrf_k + 1.0)
    ranked = sorted(
        fused.values(),
        key=lambda item: (-item[1], item[0].chunk_id),
    )
    return [
        replace(result, score=min(1.0, score / maximum_score)) for result, score in ranked[:limit]
    ]
