"""BM25 keyword search over one chunking strategy, with company / year filtering.

A single index is built over the whole corpus. Filters are applied by masking scores, so
filtered queries do not need a separate index per company.
"""

import re
from functools import lru_cache

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from annual_report_rag.retrieval import store
from annual_report_rag.retrieval.filters import Filters

STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was "
    "were what which with during year how did does do compare between".split()
)


def tokenize(text: str) -> list[str]:
    text = re.sub(r"(?<=\d),(?=\d)", "", text.lower())  # 1,62,990 -> 162990
    return [t for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, docs: list[Document] | tuple[Document, ...]):
        self.docs = list(docs)
        self.bm25 = BM25Okapi([tokenize(d.page_content) for d in self.docs]) if self.docs else None

    def search(self, query: str, k: int, filters: Filters | None = None):
        """Return up to k (document, score) pairs with a positive score."""
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        if filters and (filters.companies or filters.fiscal_years):
            mask = np.array([filters.matches(d) for d in self.docs])
            scores = np.where(mask, scores, -np.inf)
        top = np.argsort(-scores)[:k]
        return [(self.docs[i], float(scores[i])) for i in top if scores[i] > 0]


@lru_cache
def get_bm25(strategy: str) -> BM25Index:
    return BM25Index(store.load_chunks(strategy))
