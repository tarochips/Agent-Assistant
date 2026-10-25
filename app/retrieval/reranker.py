"""Optional local cross-encoder reranking for the fused candidate set."""

import asyncio
import math
import threading
from typing import Any, Protocol

from app.retrieval.models import SearchResult


class TextReranker(Protocol):
    model_id: str

    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> list[SearchResult]: ...


class NoOpReranker:
    """Keep the RRF order when cross-encoder reranking is disabled."""

    model_id = "none"

    async def rerank(
        self,
        _query: str,
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        return candidates


class FastEmbedCrossEncoderReranker:
    """Rerank candidates locally with a FastEmbed ONNX cross-encoder."""

    DEFAULT_MODEL = "BAAI/bge-reranker-base"

    def __init__(
        self,
        cache_dir: str,
        *,
        model_name: str = DEFAULT_MODEL,
        threads: int = 2,
    ) -> None:
        self.model_id = model_name
        self._cache_dir = cache_dir
        self._threads = threads
        self._model: Any = None
        self._model_lock = threading.Lock()

    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        if not candidates:
            return []
        return await asyncio.to_thread(self._rerank_sync, query, candidates)

    def _rerank_sync(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        model = self._load_model()
        raw_scores = list(model.rerank(query, [result.text for result in candidates]))
        if len(raw_scores) != len(candidates):
            raise ValueError("reranker returned a different number of scores than candidates")
        ranked = [
            (self._sigmoid(float(score)), result)
            for score, result in zip(raw_scores, candidates, strict=True)
        ]
        ranked.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            SearchResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                filename=result.filename,
                chunk_index=result.chunk_index,
                text=result.text,
                score=score,
            )
            for score, result in ranked
        ]

    def _load_model(self):
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                self._model = TextCrossEncoder(
                    model_name=self.model_id,
                    cache_dir=self._cache_dir,
                    threads=self._threads,
                    lazy_load=True,
                )
        return self._model

    @staticmethod
    def _sigmoid(score: float) -> float:
        if score >= 0:
            return 1.0 / (1.0 + math.exp(-min(score, 60.0)))
        exp_score = math.exp(max(score, -60.0))
        return exp_score / (1.0 + exp_score)
