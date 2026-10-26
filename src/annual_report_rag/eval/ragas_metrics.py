"""RAGAS metrics (v0.4 collections API) judged by a Groq model.

Groq exposes an OpenAI-compatible endpoint, so the RAGAS judge is an AsyncOpenAI client pointed at
Groq. Embeddings for answer relevancy reuse the local FastEmbed model.
"""

import asyncio
from functools import lru_cache

from annual_report_rag.config import get_settings

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
GROQ_OPENAI_URL = "https://api.groq.com/openai/v1"


@lru_cache
def _judge():
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    s = get_settings()
    client = AsyncOpenAI(api_key=s.groq_api_key, base_url=GROQ_OPENAI_URL, max_retries=8)
    return llm_factory(s.judge_model, client=client, temperature=0.0, max_tokens=4096)


@lru_cache
def _embeddings():
    from ragas.embeddings.base import BaseRagasEmbedding

    from annual_report_rag.retrieval.models import get_embeddings

    class FastEmbedRagas(BaseRagasEmbedding):
        def embed_text(self, text, **kwargs):
            return get_embeddings().embed_query(text)

        async def aembed_text(self, text, **kwargs):
            return self.embed_text(text)

        def embed_texts(self, texts, **kwargs):
            return get_embeddings().embed_documents(list(texts))

        async def aembed_texts(self, texts, **kwargs):
            return self.embed_texts(texts)

    return FastEmbedRagas()


@lru_cache
def _metrics():
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    llm = _judge()
    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=_embeddings(), strictness=2),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }


async def ascore(question: str, answer: str, contexts: list[str], reference: str) -> dict:
    m = _metrics()
    contexts = contexts or ["(no context retrieved)"]
    calls = {
        "faithfulness": m["faithfulness"].ascore(
            user_input=question, response=answer, retrieved_contexts=contexts
        ),
        "answer_relevancy": m["answer_relevancy"].ascore(user_input=question, response=answer),
        "context_precision": m["context_precision"].ascore(
            user_input=question, reference=reference, retrieved_contexts=contexts
        ),
        "context_recall": m["context_recall"].ascore(
            user_input=question, reference=reference, retrieved_contexts=contexts
        ),
    }
    scores = {}
    # sequential on purpose: Groq's free tier allows few tokens per minute
    for name, call in calls.items():
        try:
            scores[name] = float((await call).value)
        except Exception as exc:
            print(f"  {name} failed: {type(exc).__name__}: {str(exc)[:160]}")
            scores[name] = None
    return scores


def score(question: str, answer: str, contexts: list[str], reference: str) -> dict:
    return asyncio.run(ascore(question, answer, contexts, reference))
