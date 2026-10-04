"""PDF -> ordered blocks (headings, body text, tables), each tagged with its page number.

PyMuPDF gives us font sizes, so headings are found by size instead of guessing from text.
Tables are pulled out on their own and rewritten as flat "column: value" lines, which is the
format the source guide itself recommends for RAG.
"""
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pymupdf

# (minimum font size, heading level); first match wins
HEADING_SIZES = ((19.5, 1), (17.5, 2), (15.5, 3))
MIN_BODY_SIZE = 11.0            # smaller text is the running page header/footer
FRONT_MATTER_END = "table of contents"   # drop every page up to and including this one


@dataclass
class Block:
    page: int        # 1-indexed physical PDF page
    kind: str        # "heading" | "text" | "table"
    text: str
    level: int = 0   # heading level, 1 = top


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)          # "ﬁ" -> "fi"
    text = text.replace("­", "")                   # soft hyphen
    text = re.sub(r"-\n(?=\S)", "-", text)              # "flat-\nlevel" -> "flat-level"
    text = re.sub(r"\.{4,}", " ", text)                 # table-of-contents dot leaders
    return re.sub(r"[ \t]+", " ", text).strip()


def table_to_flat_text(rows: list[list]) -> str:
    """Rewrite a table as one line per row: 'Header: value; Header: value'."""
    rows = [[clean_text(str(c)) if c is not None else "" for c in r] for r in rows if any(r)]
    if len(rows) < 2:
        return "\n".join(" | ".join(r) for r in rows)
    header, body = rows[0], rows[1:]
    lines = []
    for row in body:
        pairs = [f"{h}: {v}" for h, v in zip(header, row) if v]
        if pairs:
            lines.append("; ".join(pairs))
    return "\n".join(lines)


def _heading_level(size: float) -> int:
    for min_size, level in HEADING_SIZES:
        if size >= min_size:
            return level
    return 0


def _join_lines(lines: list[str]) -> str:
    """Join the lines of one paragraph; a bullet starts a new line."""
    out = ""
    for line in lines:
        if not out:
            out = line
        elif line.startswith("•"):
            out += "\n" + line
        elif out.endswith("-"):
            out += line
        else:
            out += " " + line
    return out


def _page_tables(page) -> list:
    try:
        return list(page.find_tables().tables)
    except Exception:  # older PyMuPDF, or a page it cannot analyse
        return []


def _in_any(bbox, rects) -> bool:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in rects)


def parse_pdf(path: str | Path, front_matter_end: str | None = FRONT_MATTER_END) -> list[Block]:
    doc = pymupdf.open(str(path))
    blocks: list[Block] = []

    for page_index, page in enumerate(doc):
        page_no = page_index + 1
        tables = _page_tables(page)
        table_rects = [t.bbox for t in tables]

        page_blocks: list[Block] = []
        for raw in page.get_text("dict")["blocks"]:
            if raw.get("type") != 0:       # skip images
                continue
            para: list[str] = []
            for line in raw["lines"]:
                if _in_any(line["bbox"], table_rects):
                    continue
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                size = max(s["size"] for s in spans)
                text = clean_text("".join(s["text"] for s in line["spans"]))
                if not text or size < MIN_BODY_SIZE:
                    continue
                level = _heading_level(size)
                if level:
                    if para:
                        page_blocks.append(Block(page_no, "text", _join_lines(para)))
                        para = []
                    prev = page_blocks[-1] if page_blocks else None
                    if prev and prev.kind == "heading" and prev.level == level:
                        # a heading that wrapped onto a second line
                        prev.text += text if prev.text.endswith("-") else " " + text
                    else:
                        page_blocks.append(Block(page_no, "heading", text, level))
                else:
                    para.append(text)
            if para:
                page_blocks.append(Block(page_no, "text", _join_lines(para)))

        for table in tables:
            flat = table_to_flat_text(table.extract())
            if flat:
                page_blocks.append(Block(page_no, "table", flat))
        blocks.extend(page_blocks)

    if front_matter_end:
        for b in blocks:
            if b.kind == "heading" and b.text.lower() == front_matter_end:
                blocks = [x for x in blocks if x.page > b.page]
                break
    return blocks
