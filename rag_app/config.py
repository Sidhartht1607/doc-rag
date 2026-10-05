"""Settings, read once from environment variables (and .env if present)."""
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# the one sentence the agent must use when the document does not contain the answer
ABSTAIN_PHRASE = "I could not find this in the document."


def _env_float(name: str) -> float | None:
    value = os.getenv(name)
    return float(value) if value else None


@dataclass(frozen=True)
class Settings:
    document_path: Path
    index_dir: Path
    embed_model: str = "sentence-transformers/all-mpnet-base-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    prepend_heading: bool = True       # embed "section heading + text" so a chunk knows its topic
    top_k: int = 4                     # passages handed to the LLM
    rerank_candidates: int = 20        # hybrid results passed to the cross-encoder
    rrf_k: int = 60                    # reciprocal rank fusion constant
    default_mode: str = "hybrid_rerank"
    # tool-level abstention gate: if the best cross-encoder score is below this, the tool says
    # "no relevant passages". None = gate off. Calibrate it with `python -m rag_app.evaluate calibrate`.
    min_rerank_score: float | None = None
    llm_provider: str = "groq"         # "groq" or "openai"
    llm_model: str = "qwen/qwen3.8-27b"          # Groq model
    openai_model: str = "gpt-5-nano"
    openai_reasoning_effort: str = "low"         # gpt-5 models reason by default; "minimal" is cheapest
    max_agent_steps: int = 6           # LangGraph steps per question (2 searches + final answer)
    # optional guardrails, all off by default (see guardrails.py)
    pii_redaction: bool = False        # PII_REDACTION=1: redact the question before the LLM and the traces
    faithfulness_check: bool = False   # FAITHFULNESS_CHECK=1: a second model scores each answer's grounding
    judge_provider: str = "groq"       # must differ from the answerer; "groq" or "openai"
    judge_model: str = "llama-3.3-70b-versatile"
    min_faithfulness: float | None = None   # below this the answer is replaced by the abstain phrase; set after calibrating


def get_settings() -> Settings:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:  # python-dotenv is optional at runtime
        pass
    return Settings(
        document_path=Path(os.getenv("DOCUMENT_PATH", ROOT / "document.pdf")),
        index_dir=Path(os.getenv("INDEX_DIR", ROOT / "data" / "index")),
        # explicit LLM_PROVIDER wins; otherwise use OpenAI whenever its key is set, else Groq
        llm_provider=os.getenv("LLM_PROVIDER") or ("openai" if os.getenv("OPENAI_API_KEY") else "groq"),
        llm_model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5-nano"),
        openai_reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT", "low"),
        min_rerank_score=_env_float("MIN_RERANK_SCORE"),
        pii_redaction=os.getenv("PII_REDACTION") == "1",
        faithfulness_check=os.getenv("FAITHFULNESS_CHECK") == "1",
        judge_provider=os.getenv("JUDGE_PROVIDER", "groq"),
        judge_model=os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile"),
        min_faithfulness=_env_float("MIN_FAITHFULNESS"),
    )
