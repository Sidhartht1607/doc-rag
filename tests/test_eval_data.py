import re
import unicodedata

import pytest

from rag_app.chunk import build_chunks
from rag_app.evaluate import load_questions
from rag_app.parse import parse_pdf


def norm(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).lower()


def test_question_set_shape():
    qs = load_questions()
    assert 40 <= len(qs) <= 60
    assert len({q["id"] for q in qs}) == len(qs)
    unanswerable = [q for q in qs if not q["evidence"]]
    assert 8 <= len(unanswerable) <= 12
    assert {q["type"] for q in qs} == {"lookup", "paraphrase", "multi_step", "unanswerable"}
    assert all(q["reference_answer"] for q in qs)


@pytest.mark.slow
def test_every_evidence_phrase_exists_in_a_chunk(real_settings):
    chunks = build_chunks(parse_pdf(real_settings.document_path))
    texts = [norm(c.text) for c in chunks]
    missing = [(q["id"], e) for q in load_questions() for e in q["evidence"] if not any(norm(e) in t for t in texts)]
    assert not missing, f"evidence phrases not found in any chunk: {missing}"
