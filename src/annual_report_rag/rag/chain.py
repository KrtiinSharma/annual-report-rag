"""Hybrid RAG: plan → retrieve → answer with inline [n] citations."""

import time
from dataclasses import asdict, dataclass

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from annual_report_rag.rag.citations import (
    Citation,
    build_citations,
    format_context,
    normalise_citations,
)
from annual_report_rag.retrieval.filters import Filters
from annual_report_rag.retrieval.hybrid import RetrievalOptions, retrieve
from annual_report_rag.retrieval.models import get_llm

SYSTEM = """You are a financial analyst answering questions about Indian companies' annual reports.

Rules:
- Use ONLY the numbered context passages. Cite every fact with its passage number,
  e.g. [2] or [1][4].
- Quote figures exactly as reported, with units (₹ crore, ₹ million, US$, %). Say whether a
  figure is standalone or consolidated when the passage says so.
- For "revenue", use "Revenue from operations" (or "Revenues" in a Form 20-F), not "Total income".
- When asked for growth and both years' figures are available, compute it and show the arithmetic.
  Use only figures printed in the passages; never back-calculate a missing year from a growth rate.
- When comparing companies, give each company's figures, then the comparison. A short Markdown table
  helps for multi-company or multi-year comparisons.
- If the passages do not contain the answer, say what is missing. Do not use outside knowledge.
- Be concise: lead with the answer, then the supporting detail."""

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM), ("human", "Context passages:\n\n{context}\n\nQuestion: {question}")]
)


@dataclass
class AnswerResult:
    question: str
    answer: str
    citations: list[Citation]
    plan: list[dict]
    timings: dict[str, float]
    mode: str = "hybrid"

    @property
    def contexts(self) -> list[str]:
        return [c.snippet for c in self.citations]

    def to_dict(self) -> dict:
        return asdict(self)


@traceable(name="hybrid_rag")
def answer_question(
    question: str, filters: Filters | None = None, opts: RetrievalOptions | None = None
) -> AnswerResult:
    filters = filters or Filters()
    opts = opts or RetrievalOptions()
    started = time.perf_counter()
    result = retrieve(question, filters, opts)
    docs = [h.doc for h in result.hits]

    t = time.perf_counter()
    if docs:
        chain = PROMPT | get_llm("answer") | StrOutputParser()
        answer = chain.invoke({"context": format_context(docs), "question": question})
        answer = normalise_citations(answer)
    else:
        answer = "I could not find anything relevant in the selected reports."
    timings = {**result.timings, "generate": time.perf_counter() - t}
    timings["total"] = time.perf_counter() - started

    citations = build_citations(
        answer, docs, [h.score for h in result.hits], [h.sources for h in result.hits]
    )
    plan = [
        {"queries": tg.queries, "company": tg.company, "fiscal_year": tg.fiscal_year}
        for tg in result.plan.targets
    ]
    return AnswerResult(
        question, answer, citations, plan, {k: round(v, 3) for k, v in timings.items()}
    )
