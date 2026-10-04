"""BM25 keyword search, written out (about 25 lines of maths) so there is no extra dependency."""
import math
import re
import unicodedata
from collections import Counter

# generic English stopwords only; nothing tuned to this document or its questions
STOPWORDS = frozenset(
    "a an and are as at be but by for from has have how i in is it its of on or that the their "
    "there these this those to was were what when where which who why will with would you your "
    "do does did can could should about into than then so if not no".split()
)


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    tokens = []
    for word in re.findall(r"[a-z0-9]+", text):
        if word in STOPWORDS:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]               # crude plural folding: "lists" -> "list"
        tokens.append(word)
    return tokens


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in documents]
        self.tf = [Counter(d) for d in self.docs]
        self.avg_len = sum(len(d) for d in self.docs) / max(len(self.docs), 1)
        n = len(self.docs)
        doc_freq = Counter(t for d in self.docs for t in set(d))
        self.idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in doc_freq.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = []
        for doc, tf in zip(self.docs, self.tf):
            norm = 1 - self.b + self.b * len(doc) / (self.avg_len or 1)
            out.append(
                sum(
                    self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * norm)
                    for t in terms
                    if t in tf
                )
            )
        return out

    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        """(document index, score) for the best k documents; zero-score documents are dropped."""
        scored = sorted(enumerate(self.scores(query)), key=lambda x: -x[1])
        return [(i, s) for i, s in scored[:k] if s > 0]
