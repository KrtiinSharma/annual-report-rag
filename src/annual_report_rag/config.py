"""Runtime settings, read from environment variables or a local `.env` file."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # LLMs (Groq)
    groq_api_key: str = ""
    answer_model: str = "openai/gpt-oss-120b"
    fast_model: str = "openai/gpt-oss-20b"
    judge_model: str = "qwen/qwen3.8-27b"

    # LangSmith (picked up by langchain automatically when set in the environment)
    langsmith_api_key: str = ""
    langsmith_tracing: bool = False
    langsmith_project: str = "annual-report-analyst"

    # Models that run locally on CPU
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # Retrieval
    chunk_strategy: str = "structure"  # "fixed" | "structure"
    candidates_per_retriever: int = 20
    rerank_top_n: int = 6
    max_context_chunks: int = 12
    rrf_k: int = 60

    # Paths
    data_dir: Path = ROOT / "data"

    # API
    allowed_origins: str = "http://localhost:5173"
    rate_limit_per_minute: int = 10

    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "pdfs"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def tree_dir(self) -> Path:
        return self.data_dir / "trees"

    @property
    def results_dir(self) -> Path:
        return ROOT / "docs" / "results"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    # LangChain and the LangSmith SDK read tracing settings from os.environ, not from .env.
    if settings.langsmith_api_key:
        os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
        os.environ.setdefault("LANGSMITH_TRACING", str(settings.langsmith_tracing).lower())
    return settings
