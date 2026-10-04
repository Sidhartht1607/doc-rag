import pymupdf
import pytest

from rag_app.chunk import build_chunks
from rag_app.parse import parse_pdf, table_to_flat_text


def _write(page, y, text, size):
    page.insert_text((60, y), text, fontsize=size)
    return y + size + 8


@pytest.fixture
def synthetic_pdf(tmp_path):
    """3 pages: a front-matter page, then 'Intro > Details' running over two pages, each with a tiny running header."""
    doc = pymupdf.open()
    front = doc.new_page()                                  # fill each page before adding the next one
    _write(front, 80, "Table of Contents", 20)
    _write(front, 120, "Intro ........ 1", 12)
    page2 = doc.new_page()
    page2.insert_text((60, 30), "Running page header", fontsize=8)
    y = _write(page2, 80, "Intro", 20)
    y = _write(page2, y, "Intro body sentence.", 12)
    y = _write(page2, y, "Details", 18)
    for i in range(30):                                    # long enough to reach the bottom of page 2
        if y < 780:
            y = _write(page2, y, f"Detail line {i} about numbered lists.", 12)
    page3 = doc.new_page()
    page3.insert_text((60, 30), "Running page header", fontsize=8)
    _write(page3, 80, "Tail text that continues the Details section.", 12)
    path = tmp_path / "synthetic.pdf"
    doc.save(str(path))
    return path


def test_front_matter_header_and_headings(synthetic_pdf):
    blocks = parse_pdf(synthetic_pdf)
    assert all(b.page >= 2 for b in blocks), "table-of-contents page should be dropped"
    assert not any("Running page header" in b.text for b in blocks)
    headings = [(b.level, b.text) for b in blocks if b.kind == "heading"]
    assert headings == [(1, "Intro"), (2, "Details")]


def test_chunks_carry_section_and_pages(synthetic_pdf):
    chunks = build_chunks(parse_pdf(synthetic_pdf), chunk_size=1000, chunk_overlap=200)
    assert chunks[0].section == "Intro" and chunks[0].page == 2
    details = [c for c in chunks if c.section == "Intro > Details"]
    assert details, "section path should join the heading levels"
    assert any(3 in c.pages for c in details), "a chunk that runs onto page 3 must say so"
    assert all(c.page >= 2 and c.section for c in chunks)


def test_sections_are_split_before_size(synthetic_pdf):
    chunks = build_chunks(parse_pdf(synthetic_pdf), chunk_size=5000, chunk_overlap=0)
    assert [c.section for c in chunks] == ["Intro", "Intro > Details"]    # never merged across a heading


def test_table_becomes_flat_lines():
    rows = [["Change", "Description", "Date"], ["Initial publication", "-", "July 24, 2025"]]
    assert table_to_flat_text(rows) == "Change: Initial publication; Description: -; Date: July 24, 2025"


def test_table_in_pdf_is_a_separate_chunk(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    _write(page, 80, "Intro", 20)
    x0, y0, w, h = 60, 140, 120, 28
    cells = [["Name", "Value"], ["alpha", "1"], ["beta", "2"]]
    for r, row in enumerate(cells):
        for c, text in enumerate(row):
            rect = pymupdf.Rect(x0 + c * w, y0 + r * h, x0 + (c + 1) * w, y0 + (r + 1) * h)
            page.draw_rect(rect, color=(0, 0, 0), width=1)
            page.insert_text((rect.x0 + 6, rect.y0 + 18), text, fontsize=12)
    path = tmp_path / "table.pdf"
    doc.save(str(path))
    chunks = build_chunks(parse_pdf(path, front_matter_end=None))
    tables = [c for c in chunks if c.kind == "table"]
    assert tables and "Name: alpha; Value: 1" in tables[0].text
    assert not any(c.kind == "text" and "alpha" in c.text for c in chunks), "table text must not leak into prose"
