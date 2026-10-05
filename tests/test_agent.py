from langchain_core.messages import AIMessage

from rag_app.agent import ALREADY_SEARCHED, Agent, cited_pages, split_citations
from rag_app.config import ABSTAIN_PHRASE, get_settings
from dataclasses import replace

from conftest import ScriptedChat, StubRetriever, make_hit, tool_call


def agent_with(script, hits=None, **settings_changes):
    settings = replace(get_settings(), **settings_changes)
    return Agent(StubRetriever(hits, settings), ScriptedChat(script=script), settings)


def test_passage_label_is_not_read_as_a_page():
    assert cited_pages("A [S4 p.10]. B [S3 p.13].") == [10, 13]


def test_split_citations_separates_verified_from_invented_pages():
    assert split_citations("x [p.12] y [p.14] z [S1 p.10]", [[10], [14, 15]]) == ([14, 10], [12])


def test_cited_pages_parses_brackets():
    assert cited_pages("A [p.12]. B [p. 14, p.15]. Not a cite [1]. Again [p.12].") == [12, 14, 15]


def test_answer_with_verified_citation():
    agent = agent_with([tool_call("numbered lists"), AIMessage(content="Numbers must be sequential [p.12].")],
                       hits=[make_hit(page=12, text="Ensure that each list item is numbered sequentially.")])
    result = agent.ask("Why should numbered lists be sequential?")
    assert not result.abstained and result.status == "ok"
    assert [c["page"] for c in result.citations] == [12]
    assert "numbered sequentially" in result.citations[0]["quote"]
    assert result.unverified_citation_pages == [] and result.queries == ["numbered lists"]
    assert result.llm_calls == 2


def test_citation_to_a_page_that_was_not_retrieved_is_flagged():
    agent = agent_with([tool_call("x"), AIMessage(content="Made up [p.99].")], hits=[make_hit(page=12)])
    result = agent.ask("question here")
    assert result.citations == [] and result.unverified_citation_pages == [99]


def test_abstains_with_the_exact_phrase():
    agent = agent_with([tool_call("refund policy"), AIMessage(content=ABSTAIN_PHRASE)])
    result = agent.ask("What is the refund policy?")
    assert result.abstained and result.citations == []


def test_repeated_query_is_blocked_not_searched_twice():
    agent = agent_with([tool_call("same query", "1"), tool_call("Same  Query", "2"),
                        AIMessage(content=ABSTAIN_PHRASE)])
    result = agent.ask("anything at all")
    assert agent.retriever.queries == ["same query"], "the duplicate must not reach the retriever"
    assert result.queries == ["same query", "Same  Query"]


def test_score_gate_returns_no_passages():
    agent = agent_with([tool_call("off topic"), AIMessage(content=ABSTAIN_PHRASE)],
                       hits=[make_hit(score=-9.0)], min_rerank_score=0.0)
    result = agent.ask("off topic question")
    assert result.retrieved == [] and result.abstained


def test_endless_searching_hits_the_step_cap_and_abstains():
    script = [tool_call(f"query {i}", str(i)) for i in range(50)]
    agent = agent_with(script, max_agent_steps=6)
    result = agent.ask("never answerable")
    assert result.status == "step_limit" and result.abstained
    assert len(agent.retriever.queries) <= 3, "step cap should stop the loop after a few searches"


def test_faithfulness_judge_scores_the_answer_and_lists_unsupported_claims():
    from rag_app.guardrails import Claim, Verdict
    from test_guardrails import FakeJudge

    verdict = Verdict(claims=[Claim(claim="Lists must be sequential.", supported=True),
                              Claim(claim="Lists must be red.", supported=False)])
    agent = agent_with([tool_call("lists"), AIMessage(content="Sequential [p.12]. Red [p.12].")],
                       hits=[make_hit(page=12)], faithfulness_check=True)
    agent._judge = FakeJudge(verdict)
    result = agent.ask("Why should numbered lists be sequential?")
    assert result.faithfulness == 0.5 and result.unsupported_claims == ["Lists must be red."]
    assert result.status == "ok"


def test_low_faithfulness_replaces_the_answer_only_when_a_floor_is_set():
    from rag_app.guardrails import Claim, Verdict
    from test_guardrails import FakeJudge

    bad = Verdict(claims=[Claim(claim="invented", supported=False)])
    agent = agent_with([tool_call("lists"), AIMessage(content="Invented [p.12].")], hits=[make_hit(page=12)],
                       faithfulness_check=True, min_faithfulness=0.5)
    agent._judge = FakeJudge(bad)
    result = agent.ask("Why should numbered lists be sequential?")
    assert result.abstained and result.answer == ABSTAIN_PHRASE and result.status == "low_faithfulness"


def test_judge_is_skipped_when_the_flag_is_off():
    agent = agent_with([tool_call("lists"), AIMessage(content="Fine [p.12].")], hits=[make_hit(page=12)])
    assert agent.ask("Why should numbered lists be sequential?").faithfulness is None
