"""Cross-encoder reranking (ms-marco MiniLM via FastEmbed, runs on CPU)."""

from langchain_core.documents import Document
from langsmith import traceable

from annual_report_rag.retrieval.models import get_cross_encoder


@traceable(run_type="retriever", name="rerank")
def rerank(query: str, docs: list[Document], top_n: int) -> list[tuple[Document, float]]:
    if not docs:
        return []
    scores = list(get_cross_encoder().rerank(query, [d.page_content for d in docs], batch_size=32))
    ranked = sorted(zip(docs, scores, strict=True), key=lambda p: p[1], reverse=True)
    return [(d, float(s)) for d, s in ranked[:top_n]]
