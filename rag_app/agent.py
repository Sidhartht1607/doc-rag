"""LangGraph tool-calling agent: searches the document, cites pages, and abstains when it cannot answer.

The graph is the usual loop (LLM -> tool -> LLM). Three guards keep it from running away, all of
which were needed during development:
  * a step cap per question (one unanswerable question made the model search 7+ times and blow
    through the provider's token limit),
  * a repeated-query guard (one answerable question re-sent the identical search 4 times),
  * an optional "no relevant passages" gate on the reranker score (off by default, see config).
"""
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from .config import ABSTAIN_PHRASE, Settings
from .retrieve import Retriever

SYSTEM_PROMPT = f"""You answer questions about one document using the search_document tool.
1. Always call search_document before answering. Write the query as 2-6 keywords for the topic, \
leaving out filler words such as 'guide' and 'document'.
2. Answer only from the passages the tool returns. Passages are labelled like [S1 p.12]. \
After every claim, cite its page like [p.12]. Never cite a page the tool did not return.
3. If the passages do not contain the answer, reply with exactly: {ABSTAIN_PHRASE}
4. Search at most twice. If a search returns NO_RELEVANT_PASSAGES or ALREADY_SEARCHED, do not \
search again; answer from what you have or use the sentence in rule 3.
5. Keep the answer short and direct."""

NO_PASSAGES = (
    "NO_RELEVANT_PASSAGES: nothing in the document matches this query well. Do not search again. "
    f"Reply with exactly: {ABSTAIN_PHRASE}"
)
ALREADY_SEARCHED = (
    "ALREADY_SEARCHED: you have already run this search. Answer from the earlier passages or reply "
    f"with exactly: {ABSTAIN_PHRASE}"
)

_CITATION_GROUP = re.compile(r"\[([^\]]*)\]")


def cited_pages(answer: str) -> list[int]:
    """Page numbers cited in the answer as [p.12] or [p.12, p.14], in order of first appearance.

    The model sometimes copies the passage label too, as in "[S4 p.10]"; the 4 there is a rank, not a page.
    """
    pages: list[int] = []
    for group in _CITATION_GROUP.findall(answer):
        group = re.sub(r"\bS\d+\b", "", group)
        if re.search(r"\bp\.?\s*\d", group, re.I):
            pages += [int(n) for n in re.findall(r"\d+", group) if int(n) not in pages]
    return list(dict.fromkeys(pages))


def split_citations(answer: str, retrieved_pages: list[list[int]]) -> tuple[list[int], list[int]]:
    """(verified, unverified) cited pages. Verified = some retrieved chunk covers that page."""
    covered = {pg for pages in retrieved_pages for pg in pages}
    cited = cited_pages(answer)
    return [p for p in cited if p in covered], [p for p in cited if p not in covered]


@dataclass
class AskResult:
    answer: str
    abstained: bool
    citations: list[dict] = field(default_factory=list)          # verified: page was actually retrieved
    unverified_citation_pages: list[int] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    llm_calls: int = 0
    status: str = "ok"                                            # ok | step_limit
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    faithfulness: float | None = None                             # share of claims the passages support
    unsupported_claims: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _text(content) -> str:
    """Message content is a string, or a list of blocks for some providers."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


class _State(TypedDict):
    messages: Annotated[list, add_messages]
    llm_calls: int


class _SearchTool:
    """One instance per question, so concurrent requests never share a log."""

    def __init__(self, retriever: Retriever, settings: Settings):
        self.retriever, self.settings = retriever, settings
        self.calls: list[dict] = []
        self._seen: set[str] = set()

    def __call__(self, query: str) -> str:
        key = " ".join(query.lower().split())
        if key in self._seen:
            self.calls.append({"query": query, "hits": [], "note": "already_searched"})
            return ALREADY_SEARCHED
        self._seen.add(key)
        hits = self.retriever.search(query, k=self.settings.top_k)
        gate = self.settings.min_rerank_score
        if gate is not None and (not hits or hits[0].score < gate):
            self.calls.append({"query": query, "hits": [], "note": "gated"})
            return NO_PASSAGES
        self.calls.append({"query": query, "hits": hits, "note": "ok"})
        return "\n\n".join(
            f"[S{h.rank} p.{h.chunk.page}] {h.chunk.section}\n{h.chunk.text}" for h in hits
        )


class Agent:
    def __init__(self, retriever: Retriever, llm=None, settings: Settings | None = None, judge=None):
        self.retriever = retriever
        self.settings = settings or retriever.settings
        self._llm = llm
        self._judge = judge

    @property
    def judge(self):
        """The faithfulness judge: a different model from the answerer, built from JUDGE_PROVIDER / JUDGE_MODEL."""
        if self._judge is None:
            s = self.settings
            if s.judge_provider == "openai":
                from langchain_openai import ChatOpenAI

                self._judge = ChatOpenAI(model=s.judge_model, max_retries=2)
            elif s.judge_provider == "groq":
                from langchain_groq import ChatGroq

                self._judge = ChatGroq(model=s.judge_model, temperature=0, max_retries=2)
            else:
                raise ValueError(f"unknown JUDGE_PROVIDER {s.judge_provider!r}; use 'openai' or 'groq'")
        return self._judge

    @property
    def llm(self):
        if self._llm is None:
            if self.settings.llm_provider == "openai":
                from langchain_openai import ChatOpenAI

                # gpt-5 models are reasoning models: no temperature, and effort controls cost/latency
                self._llm = ChatOpenAI(model=self.settings.openai_model, max_retries=2,
                                       reasoning_effort=self.settings.openai_reasoning_effort)
            elif self.settings.llm_provider == "groq":
                from langchain_groq import ChatGroq

                self._llm = ChatGroq(model=self.settings.llm_model, temperature=0, max_retries=2)
            else:
                raise ValueError(f"unknown LLM_PROVIDER {self.settings.llm_provider!r}; use 'openai' or 'groq'")
        return self._llm

    @property
    def llm_name(self) -> str:
        s = self.settings
        return f"openai:{s.openai_model}" if s.llm_provider == "openai" else f"groq:{s.llm_model}"

    def _graph(self, search: _SearchTool):
        @tool
        def search_document(query: str) -> str:
            """Search the document and return the most relevant passages, each labelled with its page."""
            return search(query)

        llm = self.llm.bind_tools([search_document])

        def llm_call(state: _State) -> dict:
            reply = llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
            return {"messages": [reply], "llm_calls": state["llm_calls"] + 1}

        graph = StateGraph(_State)
        graph.add_node("llm_call", llm_call)
        graph.add_node("tools", ToolNode([search_document]))
        graph.add_edge(START, "llm_call")
        graph.add_conditional_edges("llm_call", tools_condition)   # -> "tools" or END
        graph.add_edge("tools", "llm_call")
        return graph.compile()

    def ask(self, question: str) -> AskResult:
        started = time.time()
        search = _SearchTool(self.retriever, self.settings)
        status, llm_calls, in_tok, out_tok = "ok", 0, 0, 0
        try:
            state = self._graph(search).invoke(
                {"messages": [HumanMessage(content=question)], "llm_calls": 0},
                {"recursion_limit": self.settings.max_agent_steps},
            )
            llm_calls = state["llm_calls"]
            last = state["messages"][-1]
            answer = _text(last.content) if isinstance(last, AIMessage) else ""
            for m in state["messages"]:
                usage = getattr(m, "usage_metadata", None) or {}
                in_tok += usage.get("input_tokens", 0)
                out_tok += usage.get("output_tokens", 0)
        except GraphRecursionError:
            status, answer = "step_limit", ABSTAIN_PHRASE      # it kept searching; treat as "cannot answer"
        answer = answer.strip() or ABSTAIN_PHRASE

        retrieved = [
            {"chunk_id": h.chunk.chunk_id, "page": h.chunk.page, "pages": h.chunk.pages,
             "section": h.chunk.section, "score": round(h.score, 3), "rank": h.rank}
            for call in search.calls for h in call["hits"]
        ]
        abstained = ABSTAIN_PHRASE.lower() in answer.lower()
        citations, unverified = [], []
        if not abstained:
            verified, unverified = split_citations(
                answer, [h.chunk.pages for call in search.calls for h in call["hits"]]
            )
            for page in verified:
                match = next(h for call in search.calls for h in call["hits"] if page in h.chunk.pages)
                citations.append({
                    "page": page, "section": match.chunk.section, "chunk_id": match.chunk.chunk_id,
                    "quote": match.chunk.text[:240],
                })
        faithfulness, unsupported = None, []
        if self.settings.faithfulness_check and not abstained:
            from .guardrails import judge_faithfulness

            passages = [h.chunk.text for call in search.calls for h in call["hits"]]
            faithfulness, claims = judge_faithfulness(self.judge, passages, answer)
            unsupported = [c["claim"] for c in claims if not c["supported"]]
            floor = self.settings.min_faithfulness
            if floor is not None and faithfulness < floor:
                answer, abstained, citations, status = ABSTAIN_PHRASE, True, [], "low_faithfulness"
        return AskResult(
            answer=answer, abstained=abstained, citations=citations,
            unverified_citation_pages=unverified, retrieved=retrieved,
            queries=[c["query"] for c in search.calls], llm_calls=llm_calls, status=status,
            latency_ms=int((time.time() - started) * 1000),
            input_tokens=in_tok, output_tokens=out_tok,
            faithfulness=faithfulness, unsupported_claims=unsupported,
        )
