"""BM25 lexical retrieval with jieba tokenization for Chinese and English text."""

import math
import re
from collections import Counter
from dataclasses import replace

import jieba

from app.retrieval.models import SearchResult

_TOKEN_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+|[a-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for word in jieba.cut(text.casefold(), HMM=False):
        tokens.extend(_TOKEN_PATTERN.findall(word))
    return tokens


class BM25Retriever:
    """Score a small or medium corpus with BM25; large corpora need an indexed backend."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be between 0 and 1")
        self._k1 = k1
        self._b = b

    def search(
        self,
        query: str,
        documents: list[SearchResult],
        *,
        top_k: int,
    ) -> list[SearchResult]:
        if top_k <= 0 or not documents:
            return []
        query_terms = set(tokenize(query))
        if not query_terms:
            return []

        tokenized_documents = [tokenize(document.text) for document in documents]
        document_count = len(tokenized_documents)
        average_length = sum(map(len, tokenized_documents)) / document_count
        if average_length == 0:
            return []

        document_frequency: Counter[str] = Counter()
        term_frequencies: list[Counter[str]] = []
        for tokens in tokenized_documents:
            frequencies = Counter(tokens)
            term_frequencies.append(frequencies)
            document_frequency.update(frequencies.keys())

        idf = {
            term: math.log1p((document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }
        ranked: list[tuple[float, int, SearchResult]] = []
        for index, (document, frequencies, tokens) in enumerate(
            zip(documents, term_frequencies, tokenized_documents, strict=True)
        ):
            length_norm = self._k1 * (1.0 - self._b + self._b * len(tokens) / average_length)
            score = sum(
                idf[term] * frequencies[term] * (self._k1 + 1.0) / (frequencies[term] + length_norm)
                for term in query_terms
                if frequencies[term]
            )
            if score > 0.0:
                ranked.append((score, index, document))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [replace(document, score=score) for score, _index, document in ranked[:top_k]]
