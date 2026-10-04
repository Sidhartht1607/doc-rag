from dataclasses import replace

import pytest

from rag_app.agent import Agent, _text
from rag_app.config import get_settings


@pytest.fixture(autouse=True)
def ignore_dotenv_file(monkeypatch):
    """get_settings() reads the developer's real .env; these tests must control the environment themselves."""
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)


def test_openai_is_chosen_when_only_its_key_is_set(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert get_settings().llm_provider == "openai"


def test_groq_is_the_fallback_without_an_openai_key(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert get_settings().llm_provider == "groq"


def test_explicit_provider_wins(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    assert get_settings().llm_provider == "groq"


def test_llm_is_built_for_each_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    base = get_settings()
    openai_agent = Agent(None, settings=replace(base, llm_provider="openai"))
    groq_agent = Agent(None, settings=replace(base, llm_provider="groq"))
    assert type(openai_agent.llm).__name__ == "ChatOpenAI" and openai_agent.llm_name.startswith("openai:")
    assert type(groq_agent.llm).__name__ == "ChatGroq" and groq_agent.llm_name.startswith("groq:")


def test_message_content_may_be_a_list_of_blocks():
    assert _text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"
