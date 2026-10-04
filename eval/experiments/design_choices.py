"""Reproduces the numbers quoted in the README section "Understanding the choices": RRF k, BM25 idf,
what the reranker fixed and broke, and chunk overlap. Retrieval only; no LLM calls.

    python eval/experiments/design_choices.py
"""
import dataclasses, math, re, unicodedata
from rag_app.config import get_settings
from rag_app.index import build_index
from rag_app.retrieve import Retriever, rrf
from rag_app.bm25 import BM25, tokenize
from rag_app.chunk import build_chunks
from rag_app.parse import parse_pdf
from rag_app.evaluate import load_questions, retrieval_metrics, normalize

qs = load_questions(); base = get_settings()
r = Retriever(build_index(base)); N = len(r.chunks)
print(f"N chunks = {N}")

print("\n=== 1. RRF worked example (one doc is rank 1 in one list only; another is rank 3 in both)")
for k in (0, 1, 10, 60, 100):
    a = 1/(k+1); b = 2/(k+3); print(f"  k={k:>3}: only-in-one-list rank1 = {a:.4f} | rank3-in-both = {b:.4f} -> winner: {'rank3-in-both' if b>a else ('tie' if abs(a-b)<1e-12 else 'rank1-in-one')}")

print("\n=== 1b. sensitivity of rrf_k and rerank candidates (40 answerable questions, k=4)")
for cand in (10, 20):
    for k in (1, 10, 60, 100):
        r.settings = dataclasses.replace(base, rrf_k=k, rerank_candidates=cand)
        h = retrieval_metrics(r, qs, "hybrid")["overall"]; hr = retrieval_metrics(r, qs, "hybrid_rerank")["overall"]
        print(f"  candidates={cand:>2} rrf_k={k:>3} | hybrid R@4 {h['recall@4']:.3f} MRR {h['mrr@4']:.3f} | hybrid_rerank R@4 {hr['recall@4']:.3f} MRR {hr['mrr@4']:.3f}")
r.settings = base

print("\n=== 2. BM25 idf on this corpus (idf = ln(1 + (N-n+0.5)/(n+0.5)))")
bm = r.bm25
for w in ["rag", "retrieval", "llm", "document", "vector", "audience", "kendra", "sequential", "numbered", "table", "hallucination", "lsh"]:
    n = sum(1 for d in bm.docs if w in set(d)); print(f"  {w:14s} in {n:>2}/{N} chunks  idf = {bm.idf.get(w, float('nan')):.3f}")

print("\n=== 3. cross-encoder: where did dense vs hybrid_rerank hit/miss (k=4)?")
def hit_ids(mode):
    out = {}
    for q in qs:
        if not q["evidence"]: continue
        hits = r.search(q["question"], k=4, mode=mode)
        txt = [normalize(h.chunk.text) for h in hits]
        out[q["id"]] = any(normalize(e) in t for e in q["evidence"] for t in txt)
    return out
d, hr = hit_ids("dense"), hit_ids("hybrid_rerank")
print("  fixed by rerank (dense missed, rerank hit):", [i for i in d if not d[i] and hr[i]])
print("  broken by rerank (dense hit, rerank missed):", [i for i in d if d[i] and not hr[i]])
print("  missed by both:", [i for i in d if not d[i] and not hr[i]])
tp = lambda ids, t: sum(1 for i in ids if i.startswith(t))
print("  paraphrase hits: dense", sum(d[i] for i in d if i.startswith('p')), "/8 | hybrid_rerank", sum(hr[i] for i in hr if i.startswith('p')), "/8")
print("  lookup hits:     dense", sum(d[i] for i in d if i.startswith('q')), "/22 | hybrid_rerank", sum(hr[i] for i in hr if i.startswith('q')), "/22")
print("\n  the reranker's own misses: was the evidence chunk among the 20 candidates, and how did it score?")
for qid in ("p02", "p06", "p08", "m02"):
    q = next(x for x in qs if x["id"] == qid)
    cand = [i for i, _ in r.hybrid(q["question"], 20)]
    ranked = r.rerank(q["question"], cand, 20)
    ev = [normalize(e) for e in q["evidence"]]
    pos = [(rk, round(sc, 2), r.chunks[i].chunk_id, r.chunks[i].page) for rk, (i, sc) in enumerate(ranked, 1) if any(e in normalize(r.chunks[i].text) for e in ev)]
    print(f"   {qid}: candidates={len(cand)}/{N}; evidence chunk(s) (rerank rank, score, id, page): {pos} | top-4 scores: {[round(s,2) for _, s in ranked[:4]]}")

print("\n=== 4. overlap: how many answers sit in 2+ chunks, and what the metric did")
blocks = parse_pdf(base.document_path)
for size, ov in [(1000, 200), (500, 200), (500, 100)]:
    s = dataclasses.replace(base, chunk_size=size, chunk_overlap=ov)
    chunks = build_chunks(blocks, size, ov); texts = [normalize(c.text) for c in chunks]
    ans = [q for q in qs if q["evidence"]]
    dup = sum(1 for q in ans if any(sum(normalize(e) in t for t in texts) >= 2 for e in q["evidence"]))
    rr = Retriever(build_index(s))
    dm = retrieval_metrics(rr, qs, "dense")["overall"]; hm = retrieval_metrics(rr, qs, "hybrid_rerank")["overall"]
    print(f"  {size}/{ov}: chunks={len(chunks):>2} | questions with a gold phrase in 2+ chunks: {dup:>2}/40 | dense R@4 {dm['recall@4']:.3f} MRR {dm['mrr@4']:.3f} R@1 {dm['recall@1']:.3f} | hybrid_rerank R@4 {hm['recall@4']:.3f} MRR {hm['mrr@4']:.3f}")
