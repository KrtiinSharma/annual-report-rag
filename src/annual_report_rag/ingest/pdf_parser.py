"""PDF → per-page elements (headings, paragraphs, tables) with PyMuPDF.

Tables are found with `page.find_tables()` and rendered as Markdown. Text that falls inside a
table's bounding box is dropped from the paragraph stream so numbers are not indexed twice.
Headings are spans noticeably larger than the document's body font size.
"""

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf

HEADING_RATIO = 1.25
MAX_HEADING_CHARS = 120
MIN_TABLE_CELLS = 4


@dataclass
class Element:
    kind: str  # "heading" | "text" | "table"
    text: str
    size: float = 0.0


@dataclass
class ParsedPage:
    page: int  # 1-based physical page number
    elements: list[Element] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(e.text for e in self.elements)


def _clean(text: str) -> str:
    text = text.replace(" ", " ").replace("​", "")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _table_markdown(page: pymupdf.Page, table) -> str | None:
    # Read each cell with get_textbox: Table.extract() can drop spaces and reorder punctuation.
    rows = [
        [_clean(page.get_textbox(pymupdf.Rect(c))).replace("\n", " ") if c else "" for c in r.cells]
        for r in table.rows
    ]
    rows = [r for r in rows if any(r)]
    if not rows or sum(1 for r in rows for c in r if c) < MIN_TABLE_CELLS:
        return None
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _inside(bbox, boxes) -> bool:
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in boxes)


def _body_font_size(doc: pymupdf.Document, sample_pages: int = 40) -> float:
    sizes: Counter[float] = Counter()
    step = max(1, len(doc) // sample_pages)
    for i in range(0, len(doc), step):
        for block in doc[i].get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    sizes[round(span["size"], 1)] += len(span["text"].strip())
    return sizes.most_common(1)[0][0] if sizes else 10.0


def parse_page(page: pymupdf.Page, body_size: float, with_tables: bool = True) -> ParsedPage:
    placed: list[tuple[float, Element]] = []
    table_boxes = []
    if with_tables:
        try:
            for table in page.find_tables().tables:
                md = _table_markdown(page, table)
                if md:
                    table_boxes.append(tuple(table.bbox))
                    placed.append((table.bbox[1], Element("table", md)))
        except Exception:  # find_tables can fail on malformed vector graphics
            pass

    for block in page.get_text("dict", sort=True)["blocks"]:
        if block.get("type") != 0 or _inside(block["bbox"], table_boxes):
            continue
        spans = [s for line in block["lines"] for s in line["spans"] if s["text"].strip()]
        if not spans:
            continue
        text = _clean(
            "\n".join(" ".join(s["text"] for s in line["spans"]) for line in block["lines"])
        )
        if not text:
            continue
        size = max(s["size"] for s in spans)
        is_heading = size >= body_size * HEADING_RATIO and len(text) <= MAX_HEADING_CHARS
        placed.append(
            (block["bbox"][1], Element("heading" if is_heading else "text", text, round(size, 1)))
        )

    placed.sort(key=lambda p: p[0])
    return ParsedPage(page=page.number + 1, elements=[e for _, e in placed])


def parse_pdf(path: Path, with_tables: bool = True) -> list[ParsedPage]:
    with pymupdf.open(path) as doc:
        body = _body_font_size(doc)
        return [parse_page(page, body, with_tables) for page in doc]


def save_pages(pages: list[ParsedPage], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")


def load_pages(path: Path) -> list[ParsedPage]:
    pages = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            pages.append(ParsedPage(d["page"], [Element(**e) for e in d["elements"]]))
    return pages
