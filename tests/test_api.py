import pytest
from fastapi.testclient import TestClient

from rag_app.agent import AskResult
from rag_app.api import app, get_agent


class FakeAgent:
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error

    def ask(self, question):
        if self.error:
            raise self.error
        return self.result


class RateLimitError(Exception):       # the API matches on the class name, whichever SDK raised it
    pass


@pytest.fixture
def client():
    yield TestClient(app)
    app.dependency_overrides.clear()


def use(agent):
    app.dependency_overrides[get_agent] = lambda: agent


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_returns_answer_and_citations(client):
    use(FakeAgent(AskResult(
        answer="Use sequential numbers [p.12].", abstained=False,
        citations=[{"page": 12, "section": "Best practices", "chunk_id": "c020", "quote": "Ensure numbering"}],
        queries=["numbered lists"], llm_calls=2, latency_ms=5)))
    body = client.post("/ask", json={"question": "Why sequential numbering?"}).json()
    assert body["answer"].startswith("Use sequential") and body["abstained"] is False
    assert body["citations"][0]["page"] == 12 and body["queries"] == ["numbered lists"]


def test_ask_reports_abstention(client):
    use(FakeAgent(AskResult(answer="I could not find this in the document.", abstained=True)))
    body = client.post("/ask", json={"question": "What is the refund policy?"}).json()
    assert body["abstained"] is True and body["citations"] == []


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "hi"}, {"question": "x" * 1001}])
def test_ask_validates_input(client, payload):
    use(FakeAgent(AskResult(answer="x", abstained=False)))
    assert client.post("/ask", json=payload).status_code == 422


def test_rate_limit_becomes_503(client):
    use(FakeAgent(error=RateLimitError("429")))
    assert client.post("/ask", json={"question": "anything here"}).status_code == 503


def test_other_upstream_errors_become_502(client):
    use(FakeAgent(error=RuntimeError("boom")))
    assert client.post("/ask", json={"question": "anything here"}).status_code == 502
