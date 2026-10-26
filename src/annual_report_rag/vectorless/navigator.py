"""Vectorless RAG: the LLM navigates the contents tree, reads whole pages and answers.

1. Pick sections: the LLM sees every section's title, page range and summary.
2. Pick pages:    the LLM sees the page groups (with their headings) inside those sections.
3. Answer:        the full text of the chosen pages (tables included) is the context.
"""

import time
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from pydantic import BaseModel, Field

from annual_report_rag.config import get_settings
from annual_report_rag.ingest.pdf_parser import load_pages
from annual_report_rag.rag.chain import PROMPT, AnswerResult
from annual_report_rag.rag.citations import build_citations, format_context
from annual_report_rag.registry import REPORTS_BY_ID
from annual_report_rag.retrieval.models import get_llm
from annual_report_rag.vectorless.tree import Node, Tree, load_tree

MAX_SECTIONS = 3
MAX_PAGES = 6
PAGE_CHARS = 4000


class Selection(BaseModel):
    reasoning: str = Field(description="one or two sentences on where the answer should be")
    node_ids: list[str] = Field(description="ids of the chosen nodes, most relevant first")


SECTION_PROMPT = """You are looking for information in {title} ({pages} pages).
Here is its table of contents:

{toc}

Question: {question}

Choose up to {k} sections most likely to contain the answer. Financial figures usually sit in
the financial highlights, management discussion and analysis, and the financial statements
and their notes."""

PAGE_PROMPT = """You are looking for information in {title}.
These page groups are inside the sections you chose, with the headings found on each:

{groups}

Question: {question}

Choose up to {k} page groups to read in full, most relevant first."""


def _toc(nodes: list[Node]) -> str:
    lines = []
    for n in nodes:
        line = f"[{n.id}] {n.title} (pages {n.start}-{n.end})"
        if n.summary:
            line += f"\n    {n.summary}"
        elif n.headings:
            line += f"\n    headings: {'; '.join(n.headings[:8])}"
        lines.append(line)
    return "\n".join(lines)


def _groups(nodes: list[Node]) -> str:
    lines = []
    for n in nodes:
        for c in n.children:
            heads = "; ".join(c.headings) or "(no headings)"
            lines.append(f"[{c.id}] pages {c.start}-{c.end} in '{n.title}': {heads}")
    return "\n".join(lines)


@lru_cache
def _tree(report_id: str) -> Tree:
    return load_tree(report_id)


@lru_cache
def _pages(report_id: str) -> dict[int, str]:
    pages = load_pages(get_settings().processed_dir / f"{report_id}.jsonl")
    return {p.page: p.text for p in pages}


@traceable(name="select_sections")
def select_sections(tree: Tree, question: str) -> Selection:
    llm = get_llm("answer").with_structured_output(Selection)
    prompt = SECTION_PROMPT.format(
        title=tree.title, pages=tree.pages, toc=_toc(tree.nodes), question=question, k=MAX_SECTIONS
    )
    return llm.invoke(prompt)


@traceable(name="select_pages")
def select_pages(tree: Tree, sections: list[Node], question: str) -> Selection:
    llm = get_llm("answer").with_structured_output(Selection)
    k = max(1, MAX_PAGES // 2)
    prompt = PAGE_PROMPT.format(title=tree.title, groups=_groups(sections), question=question, k=k)
    return llm.invoke(prompt)


def chosen_pages(sections: list[Node], group_ids: list[str]) -> list[int]:
    groups = {c.id: c for n in sections for c in n.children}
    pages: list[int] = []
    for gid in group_ids:
        g = groups.get(gid.strip())
        if g:
            pages += [p for p in range(g.start, g.end + 1) if p not in pages]
    if not pages:  # fall back to the start of the first chosen section
        first = sections[0]
        pages = list(range(first.start, min(first.end, first.start + MAX_PAGES - 1) + 1))
    return pages[:MAX_PAGES]


@traceable(name="vectorless_rag")
def answer_vectorless(question: str, report_id: str) -> AnswerResult:
    report = REPORTS_BY_ID[report_id]
    tree = _tree(report_id)
    started = time.perf_counter()
    timings = {}

    t = time.perf_counter()
    pick = select_sections(tree, question)
    by_id = {n.id: n for n in tree.nodes}
    sections = [by_id[i.strip()] for i in pick.node_ids if i.strip() in by_id][:MAX_SECTIONS]
    sections = sections or tree.nodes[:1]
    timings["select_sections"] = time.perf_counter() - t

    t = time.perf_counter()
    page_pick = select_pages(tree, sections, question)
    pages = chosen_pages(sections, page_pick.node_ids)
    timings["select_pages"] = time.perf_counter() - t

    texts = _pages(report_id)
    docs = [
        Document(
            page_content=f"{tree.title}\n{texts.get(p, '')[:PAGE_CHARS]}",
            metadata={
                "report_id": report_id,
                "company": report.company,
                "fiscal_year": report.fiscal_year,
                "page": p,
                "chunk_type": "page",
                "section": next((s.title for s in sections if s.start <= p <= s.end), ""),
            },
        )
        for p in pages
    ]

    t = time.perf_counter()
    chain = PROMPT | get_llm("answer") | StrOutputParser()
    answer = chain.invoke({"context": format_context(docs), "question": question})
    timings["generate"] = time.perf_counter() - t
    timings["total"] = time.perf_counter() - started

    citations = build_citations(answer, docs, [1.0] * len(docs))
    plan = [
        {"step": "sections", "reasoning": pick.reasoning, "chosen": [s.title for s in sections]},
        {"step": "pages", "reasoning": page_pick.reasoning, "chosen": pages},
    ]
    return AnswerResult(
        question,
        answer,
        citations,
        plan,
        {k: round(v, 3) for k, v in timings.items()},
        mode="vectorless",
    )
