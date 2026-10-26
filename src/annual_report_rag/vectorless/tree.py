"""Build a table-of-contents tree for one report: the index for vectorless RAG.

There are no embeddings and no chunks. The report becomes a two-level tree:

    section  (from the PDF outline, or from the largest headings when there is no outline)
      └─ page group (a few consecutive pages, with the headings found on them)

Each section gets a short LLM summary. At query time an LLM reads the tree like a human reads a
contents page, picks the relevant pages, and answers from the full text of those pages.

    uv run arr-tree --report indigo_FY24
    uv run arr-tree --report indigo_FY24 --no-summaries   # skip LLM summaries (offline)
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf

from annual_report_rag.config import get_settings
from annual_report_rag.ingest.pdf_parser import ParsedPage, load_pages
from annual_report_rag.registry import REPORTS_BY_ID, company_name

MAX_SECTION_PAGES = 24
GROUP_PAGES = 4
MIN_OUTLINE_ENTRIES = 5


@dataclass
class Node:
    id: str
    title: str
    start: int  # 1-based physical pages, inclusive
    end: int
    summary: str = ""
    headings: list[str] = field(default_factory=list)
    children: list["Node"] = field(default_factory=list)


@dataclass
class Tree:
    report_id: str
    title: str
    pages: int
    source: str  # "outline" | "headings"
    nodes: list[Node]


def _outline_sections(pdf: Path, n_pages: int) -> list[tuple[str, int]]:
    with pymupdf.open(pdf) as doc:
        toc = [(title.strip(), page) for level, title, page in doc.get_toc() if level == 1]
    toc = [(t, p) for t, p in toc if t and 1 <= p <= n_pages]
    return toc if len(toc) >= MIN_OUTLINE_ENTRIES else []


def _heading_sections(pages: list[ParsedPage]) -> list[tuple[str, int]]:
    """Section starts = pages whose largest heading is in the top size tier of the document."""
    sizes = sorted({e.size for p in pages for e in p.elements if e.kind == "heading"}, reverse=True)
    if not sizes:
        return [("Report", 1)]
    threshold = sizes[len(sizes) // 4]  # top quarter of distinct heading sizes
    starts: list[tuple[str, int]] = []
    for p in pages:
        big = [e for e in p.elements if e.kind == "heading" and e.size >= threshold]
        if big and (not starts or p.page - starts[-1][1] >= 2):
            starts.append((big[0].text.replace("\n", " ")[:90], p.page))
    if not starts or starts[0][1] != 1:
        starts.insert(0, ("Cover and contents", 1))
    return starts


def _page_headings(page: ParsedPage, limit: int = 6) -> list[str]:
    seen, out = set(), []
    for e in page.elements:
        if e.kind == "heading":
            h = e.text.replace("\n", " ")[:80]
            if h.lower() not in seen:
                seen.add(h.lower())
                out.append(h)
    return out[:limit]


def build_nodes(starts: list[tuple[str, int]], pages: list[ParsedPage]) -> list[Node]:
    n = len(pages)
    by_page = {p.page: p for p in pages}
    bounds = []
    for i, (title, start) in enumerate(starts):
        end = starts[i + 1][1] - 1 if i + 1 < len(starts) else n
        if end < start:
            continue
        # split very long sections so each node stays readable
        for s in range(start, end + 1, MAX_SECTION_PAGES):
            e = min(end, s + MAX_SECTION_PAGES - 1)
            suffix = f" (pages {s}-{e})" if end - start + 1 > MAX_SECTION_PAGES else ""
            bounds.append((title + suffix, s, e))

    nodes = []
    for i, (title, start, end) in enumerate(bounds, start=1):
        node = Node(id=str(i), title=title, start=start, end=end)
        for j, s in enumerate(range(start, end + 1, GROUP_PAGES), start=1):
            e = min(end, s + GROUP_PAGES - 1)
            heads = [h for pg in range(s, e + 1) for h in _page_headings(by_page[pg])]
            node.children.append(Node(id=f"{i}.{j}", title="", start=s, end=e, headings=heads[:10]))
        node.headings = [h for c in node.children for h in c.headings][:12]
        nodes.append(node)
    return nodes


SUMMARY_PROMPT = """Summarise this section of {company}'s {year} annual report in 2 sentences.
Mention the kinds of facts and figures it contains (e.g. revenue, fleet size, risks, board).

Section title: {title} (pages {start}-{end})

{text}"""


def summarise(nodes: list[Node], pages: list[ParsedPage], report) -> None:
    from annual_report_rag.retrieval.models import get_llm

    llm = get_llm("fast")
    by_page = {p.page: p for p in pages}
    for node in nodes:
        text = "\n".join(by_page[pg].text for pg in range(node.start, node.end + 1))[:3500]
        prompt = SUMMARY_PROMPT.format(
            company=company_name(report.company),
            year=report.fiscal_year,
            title=node.title,
            start=node.start,
            end=node.end,
            text=text,
        )
        try:
            node.summary = llm.invoke(prompt).content.strip()
        except Exception as exc:
            print(f"  summary failed for node {node.id}: {exc}")
        print(f"  [{node.id}] p{node.start}-{node.end} {node.title[:50]}")


def tree_path(report_id: str) -> Path:
    return get_settings().tree_dir / f"{report_id}.json"


def build_tree(report_id: str, summaries: bool = True) -> Tree:
    s = get_settings()
    report = REPORTS_BY_ID[report_id]
    pages = load_pages(s.processed_dir / f"{report_id}.jsonl")
    starts = _outline_sections(s.pdf_dir / report.file, len(pages))
    source = "outline" if starts else "headings"
    if not starts:
        starts = _heading_sections(pages)
    nodes = build_nodes(starts, pages)
    if summaries:
        summarise(nodes, pages, report)
    title = f"{company_name(report.company)} {report.fiscal_year} annual report"
    return Tree(report_id, title, len(pages), source, nodes)


def save_tree(tree: Tree) -> Path:
    path = tree_path(tree.report_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(tree), indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def _node(d: dict) -> Node:
    return Node(**{**d, "children": [_node(c) for c in d.get("children", [])]})


def load_tree(report_id: str) -> Tree:
    d = json.loads(tree_path(report_id).read_text(encoding="utf-8"))
    return Tree(**{**d, "nodes": [_node(n) for n in d["nodes"]]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="indigo_FY24", choices=sorted(REPORTS_BY_ID))
    parser.add_argument("--no-summaries", action="store_true")
    args = parser.parse_args()
    t = time.perf_counter()
    tree = build_tree(args.report, summaries=not args.no_summaries)
    path = save_tree(tree)
    print(
        f"{len(tree.nodes)} sections from {tree.source}, {tree.pages} pages "
        f"-> {path} ({time.perf_counter() - t:.0f}s)"
    )


if __name__ == "__main__":
    main()
