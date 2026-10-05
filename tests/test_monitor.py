import pytest

pytest.importorskip("mlflow")

from rag_app.monitor import summarise  # noqa: E402


def test_summary_of_trace_tags():
    rows = [
        {"abstained": "False", "faithfulness": "1.0", "unverified_citations": "0", "pii_types": "", "latency_ms": "100"},
        {"abstained": "True", "faithfulness": "None", "unverified_citations": "1", "pii_types": "CREDIT_CARD,EMAIL_ADDRESS", "latency_ms": "300"},
    ]
    s = summarise(rows)
    assert s["requests"] == 2 and s["abstain_rate"] == 0.5 and s["mean_faithfulness"] == 1.0
    assert s["unverified_citation_share"] == 0.5 and s["pii_by_type"] == {"CREDIT_CARD": 1, "EMAIL_ADDRESS": 1}
    assert s["requests_with_pii"] == 1 and summarise([]) == {"requests": 0}
