from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = "Agent-ready RAG"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_debug: bool = False

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048

    chunk_size: int = Field(default=400, gt=0)
    chunk_overlap: int = Field(default=100, ge=0)
    retrieval_top_k: int = Field(default=4, gt=0, le=20)
    retrieval_min_score: float = Field(default=0.02, ge=0.0, le=1.0)
    history_max_messages: int = Field(default=20, ge=0, le=100)
    max_upload_bytes: int = Field(default=2 * 1024 * 1024, gt=0)

    data_dir: Path = PROJECT_ROOT / "data"
    static_dir: Path = PROJECT_ROOT / "static"

    @model_validator(mode="after")
    def validate_paths_and_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def docstore_path(self) -> Path:
        return self.data_dir / "docstore.json"

    @property
    def vectorizer_path(self) -> Path:
        return self.data_dir / "vectorizer.pkl"


@lru_cache
def get_settings() -> Settings:
    return Settings()
