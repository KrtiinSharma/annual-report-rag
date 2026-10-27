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
    fallback_model: str = "qwen/qwen3.8-27b"  # last-resort answer model (own Groq quota)
    # Answer with other models when the answer model hits its daily limit. Off for evaluation,
    # so every scored answer comes from the same model.
    llm_fallbacks: bool = True

    # RAGAS judge: Claude (Anthropic API) when the model starts with "claude", otherwise Groq
    judge_model: str = "claude-haiku-5-5"
    anthropic_api_key: str = ""

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
    # The cross-encoder is the slowest step on a 1-CPU host. Reranking the top 20 fused candidates
    # instead of 40 cuts its time ~45% at the same hit@5 / MRR; cutting passages to 800-1200
    # characters is faster still but loses hit@5 (answers often sit further down a chunk).
    rerank_candidates: int = 20  # fused candidates per search target that get reranked
    rerank_max_chars: int = 0  # characters of each candidate the reranker reads (0 = all)
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
