"""Numbered context blocks for the prompt, and parsing `[n]` citations out of answers."""

import re
from dataclasses import dataclass

from langchain_core.documents import Document

from annual_report_rag.ingest.chunking import body_of
from annual_report_rag.registry import REPORTS_BY_ID, company_name


@dataclass
class Citation:
    id: int
    report_id: str
    company: str
    company_name: str
    fiscal_year: str
    page: int
    section: str
    chunk_type: str
    snippet: str
    score: float
    source_url: str
    cited: bool
    matched_by: list[str]  # which retrievers/queries found it, e.g. ["bm25:q0#2", "vector:q1#1"]


def clean_table_text(text: str) -> str:
    """Drop the stray cells PDF table extraction leaves when a cell border cuts through a number.

    "| 18,562 | 2 |" -> "| 18,562 |", "(0.3 | 3 )" -> "(0.3)", "1.9 | 9 %" -> "1.9%"
    """
    text = re.sub(r"(\d) \| \1 \)", r"\1)", text)
    text = re.sub(r"(\d) \| \1 ?%", r"\1%", text)
    return re.sub(r"(\d) \| \1 \|", r"\1 |", text)


def format_context(docs: list[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        m = d.metadata
        where = f"{company_name(m['company'])} {m['fiscal_year']} annual report, page {m['page']}"
        if m.get("section"):
            where += f", section: {m['section']}"
        blocks.append(f"[{i}] ({where})\n{clean_table_text(body_of(d))}")
    return "\n\n---\n\n".join(blocks)


def normalise_citations(answer: str) -> str:
    """gpt-oss models cite as 【3】 or 【3†L1-L4】; rewrite those to the [3] form the UI expects."""
    return re.sub(r"【\s*(\d+(?:\s*[,–-]\s*\d+)*)[^】]*】", r"[\1]", answer)


def cited_ids(answer: str, n: int) -> set[int]:
    ids = set()
    for group in re.findall(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]", answer):
        for part in re.split(r"\s*,\s*", group):
            if re.search(r"[–-]", part):
                lo, hi = (int(x) for x in re.split(r"\s*[–-]\s*", part))
                ids.update(range(lo, hi + 1))
            else:
                ids.add(int(part))
    return {i for i in ids if 1 <= i <= n}


def build_citations(
    answer: str,
    docs: list[Document],
    scores: list[float],
    matched_by: list[list[str]] | None = None,
) -> list[Citation]:
    used = cited_ids(answer, len(docs))
    out = []
    for i, (d, score) in enumerate(zip(docs, scores, strict=True), start=1):
        m = d.metadata
        report = REPORTS_BY_ID.get(m["report_id"])
        out.append(
            Citation(
                id=i,
                report_id=m["report_id"],
                company=m["company"],
                company_name=company_name(m["company"]),
                fiscal_year=m["fiscal_year"],
                page=m["page"],
                section=m.get("section", ""),
                chunk_type=m.get("chunk_type", "text"),
                snippet=clean_table_text(body_of(d)),
                score=round(score, 4),
                source_url=report.url if report else "",
                cited=i in used,
                matched_by=matched_by[i - 1] if matched_by else [],
            )
        )
    return out
