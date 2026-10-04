"""Blocks -> chunks. Split on headings first, then size/overlap inside each section.

Every chunk keeps the page it starts on, every page it touches, and its heading path
("Documentation best practices > ..."), so answers can cite them.
"""
from dataclasses import asdict, dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .parse import Block


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page: int            # page where the chunk starts
    pages: list[int]     # every page the chunk touches
    section: str         # heading path, "H1 > H2"
    kind: str            # "text" | "table"

    def embed_text(self, prepend_heading: bool = True) -> str:
        """What gets embedded / keyword-indexed. The heading gives a short chunk its topic."""
        return f"{self.section}\n{self.text}" if prepend_heading and self.section else self.text

    def to_dict(self) -> dict:
        return asdict(self)


def build_chunks(blocks: list[Block], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    chunks: list[Chunk] = []
    stack: list[str] = []                    # heading path, one entry per level
    pieces: list[tuple[int, str]] = []       # (page, paragraph) of the current section

    def section_name() -> str:
        return " > ".join(stack)

    def add(text: str, page: int, pages: list[int], section: str, kind: str) -> None:
        chunks.append(Chunk(f"c{len(chunks):03d}", text, page, pages, section, kind))

    def flush() -> None:
        nonlocal pieces
        if not pieces:
            return
        text, spans, pos = "", [], 0
        for page, para in pieces:
            if text:
                text += "\n\n"
                pos += 2
            spans.append((pos, pos + len(para), page))
            text += para
            pos += len(para)
        for doc in splitter.create_documents([text]):
            start = doc.metadata["start_index"]
            end = start + len(doc.page_content)
            touched = sorted({pg for s, e, pg in spans if s < end and e > start}) or [pieces[0][0]]
            add(doc.page_content, touched[0], touched, section_name(), "text")
        pieces = []

    for block in blocks:
        if block.kind == "heading":
            flush()
            del stack[block.level - 1:]
            stack.extend([""] * (block.level - 1 - len(stack)))   # a skipped level stays empty
            stack.append(block.text)
        elif block.kind == "table":
            flush()                          # tables are chunked on their own, never mixed with prose
            for doc in splitter.create_documents([block.text]):
                add(doc.page_content, block.page, [block.page], section_name(), "table")
        else:
            pieces.append((block.page, block.text))
    flush()
    return chunks
