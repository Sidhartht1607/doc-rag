"""FastAPI service.   uvicorn rag_app.api:app --port 8000

    POST /ask   {"question": "..."}  ->  answer, abstained, page citations, the queries the agent ran
    GET  /health
"""
import os
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import Agent
from .config import get_settings
from .index import load_or_build_index
from .retrieve import Retriever


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class Citation(BaseModel):
    page: int
    section: str
    chunk_id: str
    quote: str


class AskResponse(BaseModel):
    answer: str
    abstained: bool
    citations: list[Citation]
    unverified_citation_pages: list[int]    # pages the model cited that the search never returned
    queries: list[str]
    status: str                              # "ok", or "step_limit" if the agent kept searching
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    faithfulness: float | None = None        # set when FAITHFULNESS_CHECK=1 and the agent answered
    unsupported_claims: list[str] = []
    pii_redacted: list[str] = []             # entity types redacted from the question (PII_REDACTION=1)


@lru_cache(maxsize=1)
def get_agent() -> Agent:
    """Load (or build) the index and the models once per process."""
    settings = get_settings()
    return Agent(Retriever(load_or_build_index(settings)), settings=settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("RAG_SKIP_WARMUP") != "1":
        get_agent()                           # first request should not pay the model-loading cost
    if os.getenv("MLFLOW_TRACKING_URI"):      # tracing is opt-in, so tests and the default image stay offline
        import mlflow

        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "doc-rag"))
        try:
            mlflow.langchain.autolog()        # one span per LLM and tool call
        except Exception:                     # tracing must never stop the service from starting
            pass
    yield


def _answer(agent: Agent, question: str, pii: list[str]) -> dict:
    """`question` is already redacted, so a trace's recorded input never holds PII."""
    result = agent.ask(question).to_dict()
    result["pii_redacted"] = pii
    return result


def _traced_answer(agent: Agent, question: str, pii: list[str]) -> dict:
    import mlflow

    @mlflow.trace(name="ask")
    def traced(agent: Agent, question: str, pii: list[str]) -> dict:
        result = _answer(agent, question, pii)
        mlflow.update_current_trace(tags={
            "abstained": str(result["abstained"]),
            "faithfulness": str(result.get("faithfulness")),
            "unverified_citations": str(len(result["unverified_citation_pages"])),
            "pii_types": ",".join(pii),
            "status": result["status"],
            "latency_ms": str(result["latency_ms"]),
            "input_tokens": str(result["input_tokens"]),
            "output_tokens": str(result["output_tokens"]),
        })
        return result

    return traced(agent, question, pii)


def create_app() -> FastAPI:
    app = FastAPI(title="Document Q&A", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest, agent: Agent = Depends(get_agent)) -> dict:
        try:
            question, pii = request.question, []
            if get_settings().pii_redaction:
                from .guardrails import redact

                question, pii = redact(question)      # redact first: traces record the function's arguments
            if os.getenv("MLFLOW_TRACKING_URI"):
                return _traced_answer(agent, question, pii)
            return _answer(agent, question, pii)
        except Exception as exc:              # the LLM provider is the usual failure
            if type(exc).__name__ == "RateLimitError":
                raise HTTPException(status_code=503, detail="The language model is rate limited. Try again shortly.")
            raise HTTPException(status_code=502, detail=f"Upstream error: {type(exc).__name__}")

    return app


app = create_app()
