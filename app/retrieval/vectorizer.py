import asyncio
import threading
from pathlib import Path
from typing import Any, Literal, Protocol

from sklearn.feature_extraction.text import HashingVectorizer

InputKind = Literal["query", "passage"]


class TextVectorizer(Protocol):
    model_id: str

    async def embed(
        self,
        texts: list[str],
        *,
        input_kind: InputKind = "passage",
    ) -> list[list[float]]: ...


class FastEmbedE5Vectorizer:
    """Generate local multilingual E5 embeddings with FastEmbed/ONNX Runtime."""

    MODEL_NAME = "intfloat/multilingual-e5-small"
    MODEL_ID = f"{MODEL_NAME}:v1"
    DIMENSIONS = 384

    def __init__(
        self,
        cache_dir: Path,
        *,
        threads: int = 2,
        model_name: str = MODEL_NAME,
    ) -> None:
        self.model_name = model_name
        self.model_id = f"{model_name}:v1"
        self._cache_dir = cache_dir
        self._threads = threads
        self._model: Any = None
        self._model_lock = threading.Lock()

    async def embed(
        self,
        texts: list[str],
        *,
        input_kind: InputKind = "passage",
    ) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._embed_sync, texts, input_kind)

    def _embed_sync(self, texts: list[str], input_kind: InputKind) -> list[list[float]]:
        prefix = "query: " if input_kind == "query" else "passage: "
        model = self._load_model()
        vectors = [vector.tolist() for vector in model.embed([prefix + text for text in texts])]
        if any(len(vector) != self.DIMENSIONS for vector in vectors):
            raise ValueError(f"{self.model_name} must return {self.DIMENSIONS}-dimension vectors")
        return vectors

    def _load_model(self):
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            from fastembed import TextEmbedding
            from fastembed.common.model_description import ModelSource, PoolingType

            try:
                TextEmbedding.add_custom_model(
                    model=self.model_name,
                    pooling=PoolingType.MEAN,
                    normalization=True,
                    sources=ModelSource(hf=self.model_name),
                    dim=self.DIMENSIONS,
                    model_file="onnx/model.onnx",
                    description="Multilingual E5 small text embeddings",
                    license="mit",
                )
            except ValueError as exc:
                if "already registered" not in str(exc).lower():
                    raise
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._model = TextEmbedding(
                model_name=self.model_name,
                cache_dir=str(self._cache_dir),
                threads=self._threads,
                lazy_load=True,
            )
        return self._model


class LocalHashVectorizer:
    """Create normalized character n-gram vectors locally without a remote API."""

    DIMENSIONS = 384
    VERSION = "local-char-hash-384-v1"
    model_id = VERSION

    def __init__(self) -> None:
        self._vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            n_features=self.DIMENSIONS,
            alternate_sign=False,
            norm="l2",
        )

    async def embed(
        self,
        texts: list[str],
        *,
        input_kind: InputKind = "passage",
    ) -> list[list[float]]:
        del input_kind
        if not texts:
            return []
        return await asyncio.to_thread(self._embed_sync, texts)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        matrix = self._vectorizer.transform(texts)
        return [row.toarray().ravel().tolist() for row in matrix]
