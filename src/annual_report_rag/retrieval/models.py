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


# Groq's free tier caps each model at 200K tokens/day. When the answer model hits its cap, answers
# come from these instead (each model has its own quota), so the site keeps working.
FALLBACKS = {"answer": ["fast", "judge"]}
# The planner has a rule-based fallback and the answer model has fallback models, so both should
# fail fast on a rate limit rather than wait it out; the RAGAS judge can afford to wait.
RETRIES = {"answer": 2, "fast": 1, "judge": 6}


@lru_cache
def get_llm(kind: str = "answer", temperature: float = 0.0):
    """kind: "answer" (final answers), "fast" (query rewriting, routing) or "judge" (RAGAS)."""
    from langchain_groq import ChatGroq

    s = get_settings()
    model = {"answer": s.answer_model, "fast": s.fast_model, "judge": s.judge_model}[kind]
    # gpt-oss models reason before answering; "low" keeps latency and token use down
    extra = {"reasoning_effort": "low"} if "gpt-oss" in model else {}
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=s.groq_api_key,
        max_retries=RETRIES[kind],
        **extra,
    )


def _with_fallbacks(kind: str, build):
    from groq import RateLimitError

    fallbacks = FALLBACKS.get(kind, []) if get_settings().llm_fallbacks else []
    others = [build(k) for k in fallbacks]
    primary = build(kind)
    return (
        primary.with_fallbacks(others, exceptions_to_handle=(RateLimitError,))
        if others
        else primary
    )


def get_chat_model(kind: str = "answer"):
    """`get_llm(kind)`, falling back to other models when it hits Groq's daily token limit."""
    return _with_fallbacks(kind, get_llm)


def get_structured_llm(kind: str, schema):
    """A chat model that returns `schema` objects, with the same rate-limit fallbacks.

    Uses Groq's JSON-schema mode: gpt-oss sometimes answers in prose when forced to call a tool.
    """
    return _with_fallbacks(
        kind, lambda k: get_llm(k).with_structured_output(schema, method="json_schema")
    )
