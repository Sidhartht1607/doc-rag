from rag_app.bm25 import BM25, tokenize


def test_tokenize_drops_stopwords_and_folds_plurals():
    assert tokenize("The lists are numbered") == ["list", "numbered"]


def test_exact_term_ranks_first():
    docs = ["cosine similarity compares vector direction", "numbered lists must be sequential", "tables hurt retrieval"]
    top = BM25(docs).top("sequential numbered lists", 3)
    assert top[0][0] == 1


def test_rare_term_beats_common_term():
    docs = ["rag rag rag rag", "rag kendra", "rag rag"]
    assert BM25(docs).top("kendra", 1)[0][0] == 1


def test_no_overlap_returns_nothing():
    assert BM25(["alpha beta", "gamma"]).top("zzz", 5) == []
