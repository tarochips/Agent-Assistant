import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from app.repositories.json_io import JsonFileStore

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_EMPTY_DOCSTORE: dict[str, Any] = {"documents": {}, "chunks": []}


class JsonDocumentRepository:
    def __init__(self, path: Path, store: JsonFileStore) -> None:
        self._path = path
        self._store = store

    def list_documents(self) -> list[dict[str, Any]]:
        docstore = self._store.load(self._path, _EMPTY_DOCSTORE)
        documents = [
            {"doc_id": doc_id, **info}
            for doc_id, info in docstore.get("documents", {}).items()
        ]
        return sorted(documents, key=lambda item: item["uploaded_at"], reverse=True)

    def add_document(self, filename: str, chunk_texts: list[str]) -> str:
        doc_id = uuid.uuid4().hex[:8]

        def add(docstore: dict[str, Any]) -> str:
            docstore.setdefault("documents", {})[doc_id] = {
                "filename": filename,
                "chunk_count": len(chunk_texts),
                "uploaded_at": datetime.now().isoformat(),
            }
            chunks = docstore.setdefault("chunks", [])
            for index, text in enumerate(chunk_texts):
                chunks.append(
                    {
                        "doc_id": doc_id,
                        "filename": filename,
                        "chunk_index": index,
                        "text": text,
                        "embedding": [],
                    }
                )
            return doc_id

        return self._store.mutate(self._path, _EMPTY_DOCSTORE, add)

    def remove_document(self, doc_id: str) -> bool:
        if not _SAFE_ID.fullmatch(doc_id):
            return False

        def remove(docstore: dict[str, Any]) -> bool:
            documents = docstore.setdefault("documents", {})
            if doc_id not in documents:
                return False
            del documents[doc_id]
            docstore["chunks"] = [
                chunk
                for chunk in docstore.setdefault("chunks", [])
                if chunk.get("doc_id") != doc_id
            ]
            return True

        return self._store.mutate(self._path, _EMPTY_DOCSTORE, remove)

    def get_all_chunks(self) -> list[dict[str, Any]]:
        docstore = self._store.load(self._path, _EMPTY_DOCSTORE)
        return docstore.get("chunks", [])

    def replace_embeddings(self, embeddings: list[list[float]]) -> None:
        def replace(docstore: dict[str, Any]) -> None:
            chunks = docstore.setdefault("chunks", [])
            if len(chunks) != len(embeddings):
                raise ValueError("chunk count changed while rebuilding the index")
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                chunk["embedding"] = embedding

        self._store.mutate(self._path, _EMPTY_DOCSTORE, replace)
