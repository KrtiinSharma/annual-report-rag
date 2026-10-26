"""Two chunking strategies, compared in `eval/compare_chunking.py`.

fixed      Recursive character splitting (1000 chars, 150 overlap) of each page's text.
           Blind to layout: a table can be cut in half and lose its header row.
structure  Walks headings, paragraphs and tables in reading order. Paragraphs are grouped under
           the latest heading up to a size budget. Tables are kept whole (large ones are split by
           rows with the header repeated), and every chunk is prefixed with its section heading.

Chunks never cross a page boundary, so every chunk cites exactly one page. Both strategies
prepend the same "<company> <FY> annual report" header, so the comparison isolates chunking.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from annual_report_rag.ingest.pdf_parser import ParsedPage
from annual_report_rag.registry import Report, company_name

FIXED_SIZE = 1000
FIXED_OVERLAP = 150
STRUCT_MAX = 1400
STRUCT_MIN = 200
TABLE_MAX = 2500

STRATEGIES = ("fixed", "structure")


def _header(report: Report, section: str = "") -> str:
    head = f"{company_name(report.company)} {report.fiscal_year} annual report"
    return f"{head} | {section}" if section else head


def _doc(report: Report, page: int, body: str, kind: str, strategy: str, section: str = ""):
    return Document(
        page_content=f"{_header(report, section)}\n{body}",
        metadata={
            "report_id": report.id,
            "company": report.company,
            "fiscal_year": report.fiscal_year,
            "page": page,
            "chunk_type": kind,
            "section": section,
            "strategy": strategy,
        },
    )


def chunk_fixed(report: Report, pages: list[ParsedPage]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=FIXED_SIZE, chunk_overlap=FIXED_OVERLAP)
    docs = []
    for p in pages:
        for piece in splitter.split_text(p.text):
            if len(piece.strip()) >= 40:
                kind = "table" if piece.lstrip().startswith("|") else "text"
                docs.append(_doc(report, p.page, piece, kind, "fixed"))
    return docs


def split_table(md: str, limit: int = TABLE_MAX) -> list[str]:
    """Split a Markdown table by rows, repeating the header row and separator in every part."""
    if len(md) <= limit:
        return [md]
    lines = md.split("\n")
    head, rows = lines[:2], lines[2:]
    parts, current = [], []
    for row in rows:
        if current and len("\n".join(head + current + [row])) > limit:
            parts.append("\n".join(head + current))
            current = []
        current.append(row)
    if current:
        parts.append("\n".join(head + current))
    return parts


class _StructureChunker:
    def __init__(self, report: Report):
        self.report = report
        self.docs: list[Document] = []
        self.buffer: list[str] = []
        self.section = ""
        self.page = 0
        self.long_splitter = RecursiveCharacterTextSplitter(chunk_size=STRUCT_MAX, chunk_overlap=0)

    def flush(self) -> None:
        body = "\n\n".join(self.buffer).strip()
        self.buffer.clear()
        if len(body) < 40:
            return
        prev = self.docs[-1].metadata if self.docs else None
        if (
            prev
            and len(body) < STRUCT_MIN
            and prev["page"] == self.page
            and prev["chunk_type"] == "text"
            and prev["section"] == self.section
        ):
            self.docs[-1].page_content += "\n\n" + body  # glue short tails onto the previous chunk
            return
        self.docs.append(_doc(self.report, self.page, body, "text", "structure", self.section))

    def add_page(self, p: ParsedPage) -> None:
        self.page = p.page
        for el in p.elements:
            if el.kind == "heading":
                self.flush()
                self.section = el.text.replace("\n", " ")[:100]
            elif el.kind == "table":
                self.flush()
                for part in split_table(el.text):
                    self.docs.append(
                        _doc(self.report, p.page, part, "table", "structure", self.section)
                    )
            elif len(el.text) > STRUCT_MAX:
                self.flush()
                for piece in self.long_splitter.split_text(el.text):
                    self.buffer.append(piece)
                    self.flush()
            else:
                if self.buffer and len("\n\n".join(self.buffer)) + len(el.text) > STRUCT_MAX:
                    self.flush()
                self.buffer.append(el.text)
        self.flush()  # chunks never cross pages


def chunk_structure(report: Report, pages: list[ParsedPage]) -> list[Document]:
    chunker = _StructureChunker(report)
    for p in pages:
        chunker.add_page(p)
    return chunker.docs


def chunk(report: Report, pages: list[ParsedPage], strategy: str) -> list[Document]:
    if strategy == "fixed":
        return chunk_fixed(report, pages)
    if strategy == "structure":
        return chunk_structure(report, pages)
    raise ValueError(f"unknown chunking strategy {strategy!r}")


def body_of(doc: Document) -> str:
    """The chunk text without the context header line."""
    return doc.page_content.split("\n", 1)[-1]
