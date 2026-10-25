"""Small offline retrieval metrics and a CLI for a labeled query set."""

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.api.dependencies import get_retriever
from app.core.database import dispose_engine


def retrieval_metrics(
    rankings: Sequence[Sequence[str]],
    relevant_chunk_ids: Sequence[set[str]],
    *,
    k: int,
) -> dict[str, float | int]:
    if k <= 0:
        raise ValueError("k must be positive")
    if len(rankings) != len(relevant_chunk_ids):
        raise ValueError("each ranking must have a corresponding relevant-id set")
    if not rankings:
        return {f"recall@{k}": 0.0, f"mrr@{k}": 0.0, f"hit_rate@{k}": 0.0, "queries": 0}

    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    hits: list[float] = []
    for ranking, relevant in zip(rankings, relevant_chunk_ids, strict=True):
        top_results = ranking[:k]
        found = set(top_results) & relevant
        recalls.append(len(found) / len(relevant) if relevant else 0.0)
        reciprocal_rank = next(
            (
                1.0 / rank
                for rank, chunk_id in enumerate(top_results, start=1)
                if chunk_id in relevant
            ),
            0.0,
        )
        reciprocal_ranks.append(reciprocal_rank)
        hits.append(1.0 if found else 0.0)

    query_count = len(rankings)
    return {
        f"recall@{k}": sum(recalls) / query_count,
        f"mrr@{k}": sum(reciprocal_ranks) / query_count,
        f"hit_rate@{k}": sum(hits) / query_count,
        "queries": query_count,
    }


def load_evaluation_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("evaluation dataset must be a JSON array")
    cases: list[dict[str, Any]] = []
    for index, case in enumerate(payload):
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("query"), str)
            or not isinstance(case.get("relevant_chunk_ids"), list)
            or not all(isinstance(item, str) for item in case["relevant_chunk_ids"])
            or not case["relevant_chunk_ids"]
        ):
            raise ValueError(f"invalid evaluation case at index {index}")
        cases.append(case)
    return cases


async def run_evaluation(dataset_path: Path, *, k: int) -> dict[str, float | int]:
    cases = load_evaluation_cases(dataset_path)
    retriever = get_retriever()
    try:
        results = [
            await retriever.search(query=case["query"], top_k=k, min_score=0.0) for case in cases
        ]
        rankings = [[result.chunk_id for result in result_list] for result_list in results]
        labels = [set(case["relevant_chunk_ids"]) for case in cases]
        return retrieval_metrics(rankings, labels, k=k)
    finally:
        await dispose_engine()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval against labeled chunk IDs."
    )
    parser.add_argument("dataset", type=Path, help="JSON array with query/relevant_chunk_ids")
    parser.add_argument("--k", type=int, default=5, help="cutoff for retrieval metrics")
    args = parser.parse_args()
    metrics = asyncio.run(run_evaluation(args.dataset, k=args.k))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
