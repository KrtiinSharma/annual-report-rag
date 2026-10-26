"""Where chunks live: a JSONL file per strategy (source for BM25) and a Chroma collection."""

import json
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document

from annual_report_rag.config import get_settings


def chunks_path(strategy: str) -> Path:
    return get_settings().index_dir / f"chunks_{strategy}.jsonl"


def chroma_dir() -> Path:
    return get_settings().index_dir / "chroma"


def collection_name(strategy: str) -> str:
    return f"reports_{strategy}"


def save_chunks(docs: list[Document], strategy: str) -> None:
    path = chunks_path(strategy)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(
                json.dumps({"text": d.page_content, "metadata": d.metadata}, ensure_ascii=False)
            )
            f.write("\n")


@lru_cache
def load_chunks(strategy: str) -> tuple[Document, ...]:
    path = chunks_path(strategy)
    if not path.exists():
        return ()
    with open(path, encoding="utf-8") as f:
        return tuple(
            Document(page_content=row["text"], metadata=row["metadata"])
            for row in map(json.loads, f)
        )


def chunk_id(doc: Document) -> str:
    m = doc.metadata
    return f"{m['report_id']}:{m['strategy']}:{m['page']}:{m['chunk_index']}"


@lru_cache
def get_vectorstore(strategy: str):
    from langchain_chroma import Chroma

    from annual_report_rag.retrieval.models import get_embeddings

    return Chroma(
        collection_name=collection_name(strategy),
        embedding_function=get_embeddings(),
        persist_directory=str(chroma_dir()),
        collection_metadata={"hnsw:space": "cosine"},
    )
