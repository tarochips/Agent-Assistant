import copy
import json
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


class JsonFileStore:
    """Small, process-local JSON store with atomic file replacement."""

    def __init__(self) -> None:
        self._lock = threading.RLock()

    def load(self, path: Path, default: T) -> T:
        with self._lock:
            return self._load_unlocked(path, default)

    def save(self, path: Path, data: Any) -> None:
        with self._lock:
            self._save_unlocked(path, data)

    def mutate(self, path: Path, default: T, operation: Callable[[T], Any]) -> Any:
        with self._lock:
            data = self._load_unlocked(path, default)
            result = operation(data)
            self._save_unlocked(path, data)
            return result

    @staticmethod
    def _load_unlocked(path: Path, default: T) -> T:
        if not path.exists():
            return copy.deepcopy(default)
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)

    @staticmethod
    def _save_unlocked(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
