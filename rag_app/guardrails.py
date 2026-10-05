"""Input and output guardrails. Both are optional: nothing here runs unless its flag is set.

  * redact(): Microsoft Presidio replaces PII in the question with tags like <CREDIT_CARD> before the LLM,
    the logs or the traces see it (PII_REDACTION=1). Its credit-card recognizer checks the Luhn checksum.
  * judge_faithfulness(): a second model splits the answer into claims and checks each against the retrieved
    passages (FAITHFULNESS_CHECK=1). It must be a different model from the answerer.
Install with `pip install -e ".[guardrails]"` and `python -m spacy download en_core_web_md`.
"""
from functools import lru_cache

from pydantic import BaseModel

PII_ENTITIES = ["CREDIT_CARD", "EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON", "IP_ADDRESS", "IBAN_CODE", "US_SSN"]
SPACY_MODEL = "en_core_web_md"


@lru_cache(maxsize=1)
def _engines():
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine

    nlp = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": SPACY_MODEL}],
    }).create_engine()
    return AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"]), AnonymizerEngine()


def redact(text: str, score_threshold: float = 0.4) -> tuple[str, list[str]]:
    """Return (redacted text, sorted entity types found)."""
    analyzer, anonymizer = _engines()
    found = analyzer.analyze(text=text, entities=PII_ENTITIES, language="en", score_threshold=score_threshold)
    # spaCy tags single capitalised tech terms ("Terraform") as people at the same 0.85 score as real names,
    # so a PERSON must be a full name (two or more words). A lone first name is therefore not redacted.
    found = [r for r in found if r.entity_type != "PERSON" or len(text[r.start:r.end].split()) >= 2]
    if not found:
        return text, []
    return anonymizer.anonymize(text=text, analyzer_results=found).text, sorted({r.entity_type for r in found})


class Claim(BaseModel):
    claim: str
    supported: bool


class Verdict(BaseModel):
    claims: list[Claim]


JUDGE_PROMPT = """You check whether an answer is supported by the passages it was written from.
Split the answer into its separate factual claims (ignore citation markers like [p.12]). For each claim, \
set supported=true only if the passages state or directly imply it; otherwise false.

PASSAGES:
{passages}

ANSWER:
{answer}"""


def judge_faithfulness(judge_llm, passages: list[str], answer: str) -> tuple[float, list[dict]]:
    """Share of the answer's claims the passages support (1.0 = fully grounded), plus the per-claim verdicts."""
    verdict = judge_llm.with_structured_output(Verdict).invoke(
        JUDGE_PROMPT.format(passages="\n\n---\n\n".join(passages), answer=answer)
    )
    if not verdict.claims:
        return 1.0, []
    score = sum(c.supported for c in verdict.claims) / len(verdict.claims)
    return score, [c.model_dump() for c in verdict.claims]
