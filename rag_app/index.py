"""Build, save and load the search index (chunks + FAISS dense vectors)."""
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np

from .chunk import Chunk, build_chunks
from .config import Settings
from .parse import parse_pdf


@lru_cache(maxsize=2)
def get_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed(texts: list[str], model_name: str) -> np.ndarray:
    vectors = get_embedder(model_name).encode(
        texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )
    return vectors.astype("float32")


@dataclass
class Index:
    chunks: list[Chunk]
    faiss_index: faiss.Index
    settings: Settings

    def embed_texts(self) -> list[str]:
        return [c.embed_text(self.settings.prepend_heading) for c in self.chunks]


def _fingerprint(settings: Settings) -> dict:
    pdf_hash = hashlib.sha256(Path(settings.document_path).read_bytes()).hexdigest()
    return {
        "pdf_sha256": pdf_hash,
        "embed_model": settings.embed_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "prepend_heading": settings.prepend_heading,
    }


def build_index(settings: Settings) -> Index:
    chunks = build_chunks(parse_pdf(settings.document_path), settings.chunk_size, settings.chunk_overlap)
    vectors = embed([c.embed_text(settings.prepend_heading) for c in chunks], settings.embed_model)
    faiss_index = faiss.IndexFlatIP(vectors.shape[1])    # inner product == cosine for normalised vectors
    faiss_index.add(vectors)
    return Index(chunks, faiss_index, settings)


def save_index(index: Index) -> None:
    out = Path(index.settings.index_dir)
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index.faiss_index, str(out / "index.faiss"))
    with open(out / "chunks.jsonl", "w") as f:
        for c in index.chunks:
            f.write(json.dumps(c.to_dict()) + "\n")
    (out / "meta.json").write_text(json.dumps(_fingerprint(index.settings), indent=2))


def load_index(settings: Settings) -> Index | None:
    """Load a saved index, or None if it is missing or was built with other settings / another PDF."""
    out = Path(settings.index_dir)
    try:
        if json.loads((out / "meta.json").read_text()) != _fingerprint(settings):
            return None
        chunks = [Chunk(**json.loads(line)) for line in (out / "chunks.jsonl").read_text().splitlines()]
        return Index(chunks, faiss.read_index(str(out / "index.faiss")), settings)
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None


def load_or_build_index(settings: Settings) -> Index:
    index = load_index(settings)
    if index is None:
        index = build_index(settings)
        save_index(index)
    return index
