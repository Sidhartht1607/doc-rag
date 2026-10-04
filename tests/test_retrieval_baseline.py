"""Regression guard: fails if retrieval quality falls below the recorded baseline."""
import json

import pytest

from rag_app.evaluate import BASELINE_PATH, load_questions, retrieval_metrics

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def baseline():
    return json.loads(BASELINE_PATH.read_text())


def test_recall_at_4_has_not_dropped(real_retriever, baseline):
    result = retrieval_metrics(real_retriever, load_questions(), baseline["mode"], baseline["k"])
    got, want = result["overall"]["recall@4"], baseline["recall@4"]
    assert got >= want, f"Recall@4 fell from {want:.3f} to {got:.3f}; misses: {result['misses']}"


def test_mrr_at_4_has_not_dropped(real_retriever, baseline):
    result = retrieval_metrics(real_retriever, load_questions(), baseline["mode"], baseline["k"])
    got, want = result["overall"]["mrr@4"], baseline["mrr@4"]
    assert got >= want - 1e-9, f"MRR@4 fell from {want:.3f} to {got:.3f}"


def test_pipeline_still_beats_the_dense_baseline(real_retriever, baseline):
    dense = retrieval_metrics(real_retriever, load_questions(), "dense", 4)["overall"]["recall@4"]
    final = retrieval_metrics(real_retriever, load_questions(), baseline["mode"], 4)["overall"]["recall@4"]
    assert final >= dense, f"{baseline['mode']} ({final:.3f}) is worse than plain dense search ({dense:.3f})"
