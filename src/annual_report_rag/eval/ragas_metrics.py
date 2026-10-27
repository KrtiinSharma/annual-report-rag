"""RAGAS metrics (v0.4 collections API).

The judge is Claude Haiku 5.5 through the Anthropic SDK by default (JUDGE_MODEL=claude-...,
ANTHROPIC_API_KEY): judging is most of an evaluation's tokens, and Groq's free tier caps each model
at 200K tokens a day. Any other JUDGE_MODEL is called on Groq through its OpenAI-compatible
endpoint. Embeddings for answer relevancy reuse the local FastEmbed model.
"""

import asyncio
from functools import lru_cache

from annual_report_rag.config import get_settings

# Context precision is left out: it makes one judge call per passage, which the Groq free tier
# cannot afford, and the page-level hit@k / MRR against reference pages measure ranking directly.
METRICS = ("faithfulness", "answer_relevancy", "context_recall")
JUDGE_MAX_CONTEXTS = 6  # passages shown to the judge (cited ones first, see run_eval)
# vectorless passes whole pages (median ~2,800 chars); a lower cap hides facts from the judge
JUDGE_MAX_CHARS = 4000
GROQ_OPENAI_URL = "https://api.groq.com/openai/v1"


@lru_cache
def _judge():
    from ragas.llms import llm_factory

    s = get_settings()
    if s.judge_model.startswith("claude"):
        from anthropic import AsyncAnthropic

        if not s.anthropic_api_key:
            raise SystemExit("JUDGE_MODEL is a Claude model: set ANTHROPIC_API_KEY in .env")
        client = AsyncAnthropic(api_key=s.anthropic_api_key, max_retries=8)
        llm = llm_factory(s.judge_model, provider="anthropic", client=client, max_tokens=4096)
        # RAGAS adds temperature/top_p by default; the Claude API rejects sampling parameters
        for key in ("temperature", "top_p", "top_k"):
            llm.model_args.pop(key, None)
        return llm

    from openai import AsyncOpenAI

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
        ContextRecall,
        Faithfulness,
    )

    llm = _judge()
    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=_embeddings(), strictness=2),
        "context_recall": ContextRecall(llm=llm),
    }


async def ascore(question: str, answer: str, contexts: list[str], reference: str) -> dict:
    m = _metrics()
    contexts = [c[:JUDGE_MAX_CHARS] for c in contexts[:JUDGE_MAX_CONTEXTS]] or ["(no context)"]
    calls = {
        "faithfulness": m["faithfulness"].ascore(
            user_input=question, response=answer, retrieved_contexts=contexts
        ),
        "answer_relevancy": m["answer_relevancy"].ascore(user_input=question, response=answer),
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
