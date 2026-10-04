"""Shared fixtures. Fast tests use stubs; tests marked `slow` need document.pdf and download models."""
import os
from types import SimpleNamespace

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

os.environ.setdefault("RAG_SKIP_WARMUP", "1")

from rag_app.chunk import Chunk                       # noqa: E402
from rag_app.config import Settings, get_settings     # noqa: E402
from rag_app.retrieve import Hit                      # noqa: E402


class ScriptedChat(BaseChatModel):
    """A chat model that replays a fixed list of messages, so agent tests never call a real LLM."""

    script: list
    cursor: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = self.script[min(self.cursor, len(self.script) - 1)]
        object.__setattr__(self, "cursor", self.cursor + 1)
        return ChatResult(generations=[ChatGeneration(message=message)])


def tool_call(query: str, call_id: str = "1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "search_document", "args": {"query": query}, "id": call_id}])


def make_hit(rank=1, page=12, text="Use sequential numbering.", section="Best practices", score=5.0) -> Hit:
    chunk = Chunk(f"c{rank:03d}", text, page, [page], section, "text")
    return Hit(chunk, score, rank)


class StubRetriever:
    """Returns canned hits; records the queries it was given."""

    def __init__(self, hits=None, settings: Settings | None = None):
        self.hits = hits if hits is not None else [make_hit()]
        self.settings = settings or get_settings()
        self.queries: list[str] = []

    def search(self, query, k=None, mode=None):
        self.queries.append(query)
        return self.hits


@pytest.fixture
def stub_settings():
    return get_settings()


@pytest.fixture(scope="session")
def real_settings():
    settings = get_settings()
    if not settings.document_path.exists():
        pytest.skip(f"{settings.document_path} not found; download the source PDF (see README)")
    return settings


@pytest.fixture(scope="session")
def real_retriever(real_settings, tmp_path_factory):
    from dataclasses import replace

    from rag_app.index import build_index
    from rag_app.retrieve import Retriever

    settings = replace(real_settings, index_dir=tmp_path_factory.mktemp("index"))
    return Retriever(build_index(settings))
