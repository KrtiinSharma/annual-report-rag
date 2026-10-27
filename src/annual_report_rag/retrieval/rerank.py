"""Cross-encoder reranking (ms-marco MiniLM via FastEmbed, runs on CPU)."""

from langchain_core.documents import Document
from langsmith import traceable

from annual_report_rag.config import get_settings
from annual_report_rag.retrieval.models import get_cross_encoder


@traceable(run_type="retriever", name="rerank")
def rerank(query: str, docs: list[Document], top_n: int) -> list[tuple[Document, float]]:
    if not docs:
        return []
    limit = get_settings().rerank_max_chars or None
    texts = [d.page_content[:limit] for d in docs]
    # small batches: ONNX keeps an arena sized to the largest batch (32 -> ~1 GB, 4 -> ~100 MB)
    scores = list(get_cross_encoder().rerank(query, texts, batch_size=4))
    ranked = sorted(zip(docs, scores, strict=True), key=lambda p: p[1], reverse=True)
    return [(d, float(s)) for d, s in ranked[:top_n]]
