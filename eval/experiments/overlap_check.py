"""Does chunk overlap inflate the metric? Rebuilds the notebook-era chunking (page by page, no headings)
at 500/200 and 500/100 and compares them on all 40 questions and on duplicate-free subsets. No LLM calls.

    python eval/experiments/overlap_check.py
"""
# Rebuild the NOTEBOOK-ERA chunking (page by page, no headings, no heading prepend) and test whether
# duplicated answers explain why 500/200 beat 500/100.
import re, unicodedata, pymupdf, numpy as np, faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_app.index import embed
from rag_app.config import get_settings
from rag_app.evaluate import load_questions, normalize
S = get_settings(); qs = [q for q in load_questions() if q["evidence"]]

def clean(t):                                   # same cleaning as RAG.ipynb
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"Copyright © 2026 Amazon Web Services.*?All rights reserved\.", "", t, flags=re.S)
    t = t.replace("AWS Prescriptive Guidance", "").replace("Writing best practices to optimize RAG applications", "")
    t = re.sub(r"\.{2,}", "", t); t = re.sub(r"\n\s*\n+", "\n", t); return t.strip()
pages = [clean(p.get_text()) for p in pymupdf.open(S.document_path)]; pages = [p for p in pages if p]

def build(size, ov):
    sp = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=ov)
    chunks = [c for p in pages for c in sp.split_text(p)]
    v = embed(chunks, S.embed_model); ix = faiss.IndexFlatIP(v.shape[1]); ix.add(v); return chunks, ix

def first_rank(chunks, ix, q, k=8):
    ids = ix.search(embed([q["question"]], S.embed_model), k)[1][0]
    texts = [normalize(chunks[i]) for i in ids]
    r = [next((n for n, t in enumerate(texts, 1) if normalize(e) in t), None) for e in q["evidence"]]
    r = [x for x in r if x]; return min(r) if r else None

def stats(ranks):
    n = len(ranks); return f"R@4 {sum(r is not None and r<=4 for r in ranks)/n:.3f} MRR@4 {sum(1/r for r in ranks if r and r<=4)/n:.3f} R@1 {sum(r==1 for r in ranks)/n:.3f}"

res = {}
for size, ov in [(500, 200), (500, 100)]:
    chunks, ix = build(size, ov); texts = [normalize(c) for c in chunks]
    dup = {q["id"]: any(sum(normalize(e) in t for t in texts) >= 2 for e in q["evidence"]) for q in qs}
    res[(size, ov)] = ({q["id"]: first_rank(chunks, ix, q) for q in qs}, dup, len(chunks))
(r200, d200, n200), (r100, d100, n100) = res[(500, 200)], res[(500, 100)]
print(f"notebook-era chunking: 500/200 -> {n200} chunks, {sum(d200.values())}/40 answers duplicated | 500/100 -> {n100} chunks, {sum(d100.values())}/40 duplicated")
ids = [q["id"] for q in qs]
print("ALL 40       500/200:", stats([r200[i] for i in ids]), "|| 500/100:", stats([r100[i] for i in ids]))
clean_ids = [i for i in ids if not d200[i] and not d100[i]]
print(f"NO-DUPLICATE questions only ({len(clean_ids)}): 500/200:", stats([r200[i] for i in clean_ids]), "|| 500/100:", stats([r100[i] for i in clean_ids]))
dirty = [i for i in ids if d200[i] or d100[i]]
print(f"duplicated questions only ({len(dirty)}):        500/200:", stats([r200[i] for i in dirty]), "|| 500/100:", stats([r100[i] for i in dirty]))
