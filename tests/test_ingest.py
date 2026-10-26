import pymupdf
import pytest

from annual_report_rag.ingest.chunking import body_of, chunk_fixed, chunk_structure, split_table
from annual_report_rag.ingest.pdf_parser import parse_pdf
from annual_report_rag.registry import Report

REPORT = Report("indigo", "FY24", "https://example.com/r.pdf")
PARAGRAPH = (
    "The Company operated a fleet of 367 aircraft and carried a record number of passengers "
    "during the year, supported by strong domestic demand. "
)


def _draw_table(page, top: float, rows: list[list[str]]) -> None:
    x0, col_w, row_h = 72, 150, 20
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            rect = pymupdf.Rect(
                x0 + c * col_w, top + r * row_h, x0 + (c + 1) * col_w, top + (r + 1) * row_h
            )
            page.draw_rect(rect, color=(0, 0, 0), width=0.8)
            page.insert_text((rect.x0 + 4, rect.y1 - 6), cell, fontsize=9)


@pytest.fixture
def sample_pdf(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 80), "Financial Highlights", fontsize=20)
    y = 110
    for _ in range(4):
        page.insert_textbox(pymupdf.Rect(72, y, 520, y + 60), PARAGRAPH, fontsize=10)
        y += 60
    _draw_table(
        page,
        380,
        [
            ["Particulars", "FY24", "FY23"],
            ["Revenue from operations", "68,904", "54,446"],
            ["Profit after tax", "8,172", "(3,059)"],
        ],
    )
    page2 = doc.new_page()
    page2.insert_text((72, 80), "Risk Management", fontsize=20)
    page2.insert_textbox(pymupdf.Rect(72, 110, 520, 400), PARAGRAPH * 3, fontsize=10)
    path = tmp_path / "sample.pdf"
    doc.save(path)
    return path


def test_parser_finds_headings_tables_and_pages(sample_pdf):
    pages = parse_pdf(sample_pdf)
    assert [p.page for p in pages] == [1, 2]
    kinds = [e.kind for e in pages[0].elements]
    assert kinds[0] == "heading" and "table" in kinds
    table = next(e.text for e in pages[0].elements if e.kind == "table")
    assert "| Revenue from operations | 68,904 | 54,446 |" in table
    # table cell text must not leak into the paragraph stream
    assert not any("68,904" in e.text for e in pages[0].elements if e.kind == "text")


def test_structure_chunks_keep_tables_whole_and_carry_sections(sample_pdf):
    docs = chunk_structure(REPORT, parse_pdf(sample_pdf))
    tables = [d for d in docs if d.metadata["chunk_type"] == "table"]
    assert len(tables) == 1
    assert "Particulars" in body_of(tables[0]) and "Profit after tax" in body_of(tables[0])
    assert tables[0].metadata["section"] == "Financial Highlights"
    assert docs[-1].metadata["page"] == 2 and docs[-1].metadata["section"] == "Risk Management"
    assert all(d.page_content.startswith("InterGlobe Aviation (IndiGo) FY24") for d in docs)


def test_fixed_chunks_stay_within_a_page(sample_pdf):
    docs = chunk_fixed(REPORT, parse_pdf(sample_pdf))
    assert {d.metadata["page"] for d in docs} == {1, 2}
    assert all(len(body_of(d)) <= 1000 for d in docs)


def test_split_table_repeats_header():
    md = "| a | b |\n| --- | --- |\n" + "\n".join(f"| r{i} | {i} |" for i in range(200))
    parts = split_table(md, limit=300)
    assert len(parts) > 1
    assert all(p.startswith("| a | b |\n| --- | --- |") for p in parts)
    assert sum(p.count("| r") for p in parts) == 200
