"""Local CPU models (FastEmbed / ONNX) and Groq chat models, created once per process."""

from functools import lru_cache

from annual_report_rag.config import ROOT, get_settings

MODEL_CACHE = ROOT / ".cache" / "fastembed"


@lru_cache
def get_embeddings():
    from langchain_community.embeddings import FastEmbedEmbeddings

    return FastEmbedEmbeddings(
        model_name=get_settings().embedding_model, cache_dir=str(MODEL_CACHE), batch_size=64
    )


@lru_cache
def get_cross_encoder():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(get_settings().reranker_model, cache_dir=str(MODEL_CACHE))


@lru_cache
def get_llm(kind: str = "answer", temperature: float = 0.0):
    """kind: "answer" (final answers), "fast" (query rewriting, routing) or "judge" (RAGAS)."""
    from langchain_groq import ChatGroq

    s = get_settings()
    model = {"answer": s.answer_model, "fast": s.fast_model, "judge": s.judge_model}[kind]
    return ChatGroq(model=model, temperature=temperature, api_key=s.groq_api_key, max_retries=6)
