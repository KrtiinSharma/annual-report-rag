"""Hybrid retrieval: BM25 + vector search → Reciprocal Rank Fusion → cross-encoder rerank.

Each search target from the query planner is retrieved separately, with its own company/year
filter, so a comparison question gets evidence for every company it mentions.
"""

import time
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langsmith import traceable

from annual_report_rag.config import get_settings
from annual_report_rag.retrieval import store
from annual_report_rag.retrieval.bm25 import get_bm25
from annual_report_rag.retrieval.filters import Filters
from annual_report_rag.retrieval.query_analysis import (
    QueryPlan,
    SearchTarget,
    plan_queries,
    target_filters,
)
from annual_report_rag.retrieval.rerank import rerank


@dataclass
class RetrievalOptions:
    strategy: str = ""
    use_bm25: bool = True
    use_vector: bool = True
    use_rerank: bool = True
    use_multi_query: bool = True
    top_n: int = 0  # chunks kept per target after reranking (0 = settings default)

    def __post_init__(self):
        s = get_settings()
        self.strategy = self.strategy or s.chunk_strategy
        self.top_n = self.top_n or s.rerank_top_n
        if not (self.use_bm25 or self.use_vector):
            raise ValueError("enable at least one of BM25 and vector search")


@dataclass
class Hit:
    doc: Document
    score: float
    sources: list[str] = field(default_factory=list)  # e.g. ["bm25#3", "vector#1"]
    target: int = 0


@dataclass
class RetrievalResult:
    plan: QueryPlan
    hits: list[Hit]
    timings: dict[str, float]


def doc_key(doc: Document) -> str:
    return store.chunk_id(doc)


def rrf(ranked_lists: dict[str, list[Document]], k: int = 60) -> list[Hit]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank)."""
    fused: dict[str, Hit] = {}
    for name, docs in ranked_lists.items():
        for rank, doc in enumerate(docs, start=1):
            hit = fused.setdefault(doc_key(doc), Hit(doc, 0.0))
            hit.score += 1.0 / (k + rank)
            hit.sources.append(f"{name}#{rank}")
    return sorted(fused.values(), key=lambda h: h.score, reverse=True)


@traceable(run_type="retriever", name="vector_search")
def vector_search(query: str, k: int, filters: Filters, strategy: str) -> list[Document]:
    return store.get_vectorstore(strategy).similarity_search(query, k=k, filter=filters.to_chroma())


@traceable(run_type="retriever", name="bm25_search")
def bm25_search(query: str, k: int, filters: Filters, strategy: str) -> list[Document]:
    return [d for d, _ in get_bm25(strategy).search(query, k, filters)]


@traceable(name="retrieve_target")
def retrieve_target(target: SearchTarget, filters: Filters, opts: RetrievalOptions) -> list[Hit]:
    s = get_settings()
    k = s.candidates_per_retriever
    lists: dict[str, list[Document]] = {}
    for i, q in enumerate(target.queries):
        if opts.use_bm25:
            lists[f"bm25:q{i}"] = bm25_search(q, k, filters, opts.strategy)
        if opts.use_vector:
            lists[f"vector:q{i}"] = vector_search(q, k, filters, opts.strategy)
    fused = rrf(lists, s.rrf_k)[: 2 * k]
    if not opts.use_rerank:
        return fused[: opts.top_n]
    by_key = {doc_key(h.doc): h for h in fused}
    reranked = rerank(target.queries[0], [h.doc for h in fused], opts.top_n)
    out = []
    for doc, score in reranked:
        hit = by_key[doc_key(doc)]
        out.append(Hit(doc, score, hit.sources))
    return out


@traceable(name="hybrid_retrieve")
def retrieve(question: str, user_filters: Filters, opts: RetrievalOptions) -> RetrievalResult:
    timings = {}
    t = time.perf_counter()
    plan = plan_queries(question, user_filters, use_llm=opts.use_multi_query)
    timings["plan"] = time.perf_counter() - t

    t = time.perf_counter()
    per_target = [
        retrieve_target(target, target_filters(target, user_filters), opts)
        for target in plan.targets
    ]
    timings["retrieve"] = time.perf_counter() - t

    # Interleave targets so every company is represented before the context budget runs out.
    seen, hits = set(), []
    limit = get_settings().max_context_chunks
    for rank in range(max((len(h) for h in per_target), default=0)):
        for ti, target_hits in enumerate(per_target):
            if rank < len(target_hits):
                hit = target_hits[rank]
                key = doc_key(hit.doc)
                if key not in seen and len(hits) < limit:
                    seen.add(key)
                    hit.target = ti
                    hits.append(hit)
    return RetrievalResult(plan, hits, timings)
