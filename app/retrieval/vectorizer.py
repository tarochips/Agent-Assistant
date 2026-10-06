import asyncio
from typing import Protocol

from sklearn.feature_extraction.text import HashingVectorizer


class TextVectorizer(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class LocalHashVectorizer:
    """Create normalized character n-gram vectors locally without a remote API."""

    DIMENSIONS = 384
    VERSION = "local-char-hash-384-v1"

    def __init__(self) -> None:
        self._vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            n_features=self.DIMENSIONS,
            alternate_sign=False,
            norm="l2",
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._embed_sync, texts)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        matrix = self._vectorizer.transform(texts)
        return [row.toarray().ravel().tolist() for row in matrix]
