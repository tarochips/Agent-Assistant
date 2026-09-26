import os
import pickle
import threading
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.repositories.documents import JsonDocumentRepository
from app.retrieval.models import SearchResult


def split_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    normalized = text.strip()
    if not normalized:
        return []

    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(normalized), step):
        chunk = normalized[start : start + chunk_size]
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(normalized):
            break
    return chunks


class TfidfRetriever:
    def __init__(
        self,
        repository: JsonDocumentRepository,
        vectorizer_path: Path,
        *,
        chunk_size: int,
        chunk_overlap: int,
        max_features: int = 10_000,
    ) -> None:
        self._repository = repository
        self._vectorizer_path = vectorizer_path
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._max_features = max_features
        self._vectorizer: TfidfVectorizer | None = None
        self._lock = threading.RLock()

    def index_document(self, filename: str, content: str) -> tuple[str, int]:
        chunk_texts = split_text(
            content,
            chunk_size=self._chunk_size,
            overlap=self._chunk_overlap,
        )
        if not chunk_texts:
            raise ValueError("document contains no indexable text")

        with self._lock:
            doc_id = self._repository.add_document(filename, chunk_texts)
            try:
                self.rebuild_index()
            except Exception:
                self._repository.remove_document(doc_id)
                self.rebuild_index()
                raise
        return doc_id, len(chunk_texts)

    def remove_document(self, doc_id: str) -> bool:
        with self._lock:
            removed = self._repository.remove_document(doc_id)
            if removed:
                self.rebuild_index()
            return removed

    def search(self, query: str, *, top_k: int, min_score: float) -> list[SearchResult]:
        chunks = self._repository.get_all_chunks()
        if not chunks:
            return []

        with self._lock:
            vectorizer = self._load_or_rebuild_vectorizer(chunks)
            query_vector = vectorizer.transform([query])
            chunk_vectors = np.asarray(
                [chunk.get("embedding", []) for chunk in chunks],
                dtype=np.float64,
            )

            if chunk_vectors.ndim != 2 or chunk_vectors.shape[1] != query_vector.shape[1]:
                vectorizer = self.rebuild_index()
                if vectorizer is None:
                    return []
                chunks = self._repository.get_all_chunks()
                query_vector = vectorizer.transform([query])
                chunk_vectors = np.asarray(
                    [chunk.get("embedding", []) for chunk in chunks],
                    dtype=np.float64,
                )

            similarities = cosine_similarity(query_vector, chunk_vectors)[0]
            ranked_indices = np.argsort(similarities)[::-1]

        results: list[SearchResult] = []
        for index in ranked_indices:
            score = float(similarities[index])
            if score < min_score:
                continue
            chunk = chunks[int(index)]
            doc_id = str(chunk["doc_id"])
            chunk_index = int(chunk["chunk_index"])
            results.append(
                SearchResult(
                    chunk_id=f"{doc_id}:{chunk_index}",
                    document_id=doc_id,
                    filename=str(chunk["filename"]),
                    chunk_index=chunk_index,
                    text=str(chunk["text"]),
                    score=score,
                )
            )
            if len(results) >= top_k:
                break
        return results

    def rebuild_index(self) -> TfidfVectorizer | None:
        with self._lock:
            chunks = self._repository.get_all_chunks()
            if not chunks:
                self._vectorizer = None
                if self._vectorizer_path.exists():
                    self._vectorizer_path.unlink()
                return None

            vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                max_features=self._max_features,
            )
            matrix = vectorizer.fit_transform([str(chunk["text"]) for chunk in chunks])
            embeddings = [row.toarray()[0].tolist() for row in matrix]
            self._repository.replace_embeddings(embeddings)
            self._save_vectorizer(vectorizer)
            self._vectorizer = vectorizer
            return vectorizer

    def _load_or_rebuild_vectorizer(
        self, chunks: list[dict[str, object]]
    ) -> TfidfVectorizer:
        if self._vectorizer is not None:
            return self._vectorizer
        if self._vectorizer_path.exists():
            try:
                with self._vectorizer_path.open("rb") as stream:
                    vectorizer = pickle.load(stream)
                expected_size = len(vectorizer.get_feature_names_out())
                actual_size = len(chunks[0].get("embedding", []))
                if expected_size == actual_size:
                    self._vectorizer = vectorizer
                    return vectorizer
            except (OSError, pickle.UnpicklingError, AttributeError, ValueError):
                pass

        rebuilt = self.rebuild_index()
        if rebuilt is None:
            raise RuntimeError("cannot build a vectorizer for an empty knowledge base")
        return rebuilt

    def _save_vectorizer(self, vectorizer: TfidfVectorizer) -> None:
        self._vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._vectorizer_path.with_suffix(".pkl.tmp")
        with temporary_path.open("wb") as stream:
            pickle.dump(vectorizer, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, self._vectorizer_path)
