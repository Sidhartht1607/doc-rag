import pytest

from rag_app.guardrails import Claim, Verdict, judge_faithfulness


class FakeJudge:
    """Stands in for a chat model: with_structured_output(...).invoke(...) returns a fixed Verdict."""

    def __init__(self, verdict):
        self.verdict = verdict

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        return self.verdict


def test_faithfulness_is_share_of_supported_claims():
    judge = FakeJudge(Verdict(claims=[Claim(claim="a", supported=True), Claim(claim="b", supported=False)]))
    score, claims = judge_faithfulness(judge, ["passage"], "a. b.")
    assert score == 0.5 and claims[1] == {"claim": "b", "supported": False}


def test_no_claims_counts_as_grounded():
    assert judge_faithfulness(FakeJudge(Verdict(claims=[])), ["p"], "ok")[0] == 1.0


@pytest.mark.slow
def test_redacts_card_and_email():
    from rag_app.guardrails import redact

    text, found = redact("My card is 4111 1111 1111 1111, email me at a@b.com")
    assert "4111" not in text and "<CREDIT_CARD>" in text and "<EMAIL_ADDRESS>" in text
    assert found == ["CREDIT_CARD", "EMAIL_ADDRESS"]


@pytest.mark.slow
def test_clean_question_is_unchanged():
    from rag_app.guardrails import redact

    q = "What are the two stages of RAG?"
    assert redact(q) == (q, [])


@pytest.mark.slow
def test_planted_pii_is_removed_and_eval_questions_are_untouched():
    import json

    from rag_app.evaluate import load_questions
    from rag_app.guardrails import redact

    root = __import__("pathlib").Path(__file__).resolve().parent.parent / "eval"
    for p in json.loads((root / "pii_questions.json").read_text()):
        text, _ = redact(p["question"])
        assert not [x for x in p["planted"] if x in text], p["id"]
    assert [q["id"] for q in load_questions() if redact(q["question"])[1]] == []
