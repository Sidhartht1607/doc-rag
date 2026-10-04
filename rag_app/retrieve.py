"""Retrieval: dense baseline -> hybrid (BM25 + dense, fused with RRF) -> cross-encoder rerank."""
from dataclasses import dataclass
from functools import lru_cache

from .bm25 import BM25
from .chunk import Chunk
from .index import Index, embed

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


@dataclass
class Hit:
    chunk: Chunk
    score: float     # cosine for dense, BM25 score, RRF score, or cross-encoder logit (by mode)
    rank: int        # 1 = best


@lru_cache(maxsize=2)
def get_reranker(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: each list votes 1/(k + rank) for every item it contains."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


class Retriever:
    def __init__(self, index: Index):
        self.index = index
        self.settings = index.settings
        self.chunks = index.chunks
        self.bm25 = BM25(index.embed_texts())
        self._reranker = None

    # --- the four modes ---------------------------------------------------------------------
    def dense(self, query: str, k: int) -> list[tuple[int, float]]:
        vector = embed([query], self.settings.embed_model)
        scores, ids = self.index.faiss_index.search(vector, min(k, len(self.chunks)))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def keyword(self, query: str, k: int) -> list[tuple[int, float]]:
        return self.bm25.top(query, k)

    def hybrid(self, query: str, k: int) -> list[tuple[int, float]]:
        depth = max(k, self.settings.rerank_candidates)
        dense_ids = [i for i, _ in self.dense(query, depth)]
        bm25_ids = [i for i, _ in self.keyword(query, depth)]
        return rrf([dense_ids, bm25_ids], self.settings.rrf_k)[:k]

    def rerank(self, query: str, candidates: list[int], k: int) -> list[tuple[int, float]]:
        if self._reranker is None:
            self._reranker = get_reranker(self.settings.rerank_model)
        texts = [self.chunks[i].embed_text(self.settings.prepend_heading) for i in candidates]
        scores = self._reranker.predict([(query, t) for t in texts], show_progress_bar=False)
        ranked = sorted(zip(candidates, (float(s) for s in scores)), key=lambda x: -x[1])
        return ranked[:k]

    # --- public entry point ---------------------------------------------------------------------
    def search(self, query: str, k: int | None = None, mode: str | None = None) -> list[Hit]:
        k = k or self.settings.top_k
        mode = mode or self.settings.default_mode
        if mode == "dense":
            pairs = self.dense(query, k)
        elif mode == "bm25":
            pairs = self.keyword(query, k)
        elif mode == "hybrid":
            pairs = self.hybrid(query, k)
        elif mode == "hybrid_rerank":
            candidates = [i for i, _ in self.hybrid(query, max(k, self.settings.rerank_candidates))]
            pairs = self.rerank(query, candidates, k)
        else:
            raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
        return [Hit(self.chunks[i], score, rank) for rank, (i, score) in enumerate(pairs, start=1)]
