from annual_report_rag.ingest.pdf_parser import Element, ParsedPage
from annual_report_rag.vectorless.navigator import _groups, _toc, chosen_pages
from annual_report_rag.vectorless.tree import _heading_sections, _page_headings, build_nodes

# one distinct sub-heading every third page (index i // 3)
WORDS = (
    "x revenue fleet profit cash debt staff fuel risk board audit tax lease "
    "cargo routes safety award brand ESG dividend outlook"
).split()


def make_pages(n=60):
    pages = []
    for i in range(1, n + 1):
        els = [Element("text", f"body text of page {i}", 10)]
        if i in (1, 5, 40):
            title = {1: "Cover", 5: "Chairman's Message", 40: "Highlights"}[i]
            els.insert(0, Element("heading", title, 24))
        if i % 3 == 0:
            els.insert(0, Element("heading", f"Topic {WORDS[i // 3]}", 14))
        pages.append(ParsedPage(i, els))
    return pages


def test_heading_sections_use_largest_headings():
    starts = _heading_sections(make_pages())
    assert [p for _, p in starts] == [1, 5, 40]


def test_heading_sections_match_statutory_parts_by_name():
    pages = make_pages()
    # body-sized headings, which the size rule alone would miss
    pages[19].elements.insert(0, Element("heading", "Report on Corporate Governance", 11))
    pages[29].elements.insert(0, Element("heading", "Standalone Balance Sheet", 11))
    starts = _heading_sections(pages)
    assert starts[:2] == [("Cover", 1), ("Chairman's Message", 5)]
    assert ("Corporate Governance Report", 20) in starts
    assert ("Standalone Financial Statements", 30) in starts
    # once the statutory parts begin, decorative big headings no longer open sections
    assert all(p != 40 for _, p in starts)


def test_long_sections_are_split_and_grouped():
    pages = make_pages()
    nodes = build_nodes([("Cover", 1), ("Directors' Report", 5), ("Notes", 40)], pages)
    # Directors' Report spans 35 pages -> split into 24 + 11
    assert [(n.start, n.end) for n in nodes] == [(1, 4), (5, 28), (29, 39), (40, 60)]
    assert all(c.end - c.start < 4 for n in nodes for c in n.children)
    assert "Topic fleet" in nodes[1].children[0].headings
    assert "[2] Directors' Report (pages 5-28)" in _toc(nodes)
    assert "[2.1] pages 5-8" in _groups(nodes[1:2])


def test_group_headings_skip_page_footers_and_keep_numbered_lines():
    pages = make_pages()
    for p in pages:  # a footer on every page, differing only by page number
        p.elements.append(Element("heading", f"{p.page} Annual Report 2023-24", 9))
    pages[5].elements.append(Element("text", "4. Operational Performance\nIndiGo flew more", 10))
    nodes = build_nodes([("Cover", 1), ("Board's Report", 5)], pages)
    headings = [h for n in nodes for c in n.children for h in c.headings]
    assert not any("Annual Report" in h for h in headings)
    assert "4. Operational Performance" in nodes[1].children[0].headings
    assert _page_headings(pages[5], limit=2) == ["Topic fleet", "6 Annual Report 2023-24"]


def test_chosen_pages_dedupes_caps_and_falls_back():
    nodes = build_nodes([("A", 1), ("B", 20)], make_pages(40))
    assert chosen_pages(nodes, ["1.1", "1.1", "2.1"]) == [1, 2, 3, 4, 20, 21]
    assert chosen_pages(nodes[1:], ["nonsense"]) == [20, 21, 22, 23, 24, 25]
