![CI](https://github.com/Sidhartht1607/doc-rag/actions/workflows/ci.yml/badge.svg)

# Document Q&A: a RAG service that cites pages and abstains

Ask questions about a PDF and get an answer **with page citations**, or a plain "I could not find this in the
document." when the answer isn't there. Retrieval is hybrid (BM25 + dense) followed by a cross-encoder rerank;
the answer comes from a LangGraph tool-calling agent; it is served with FastAPI and guarded by tests that fail
if retrieval quality drops.

The test document is *AWS Prescriptive Guidance: Writing best practices to optimize RAG applications*
(18 pages). It is AWS copyrighted material, so it is **not in this repo**. See [Setup](#setup).

> **Status, stated plainly.** Retrieval and the agent are both measured. Retrieval: Recall@4 = 0.900. Agent
> (OpenAI `gpt-5-nano`, one run): all 10 unanswerable questions abstained, none of the 40 answerable ones did,
> and 35 of 40 answers were correct under a strict 1/0 rubric. **Those answers were graded by the AI assistant
> at the owner's request, and no human has checked the grading yet**, so treat correctness as provisional.
> The design choices are explained in [Understanding the choices](#understanding-the-choices).
>
> **Since those numbers:** CI now runs the tests and the Recall@4 / MRR@4 guard on every push and builds the
> Docker image ([Continuous integration](#continuous-integration)). Optional guardrails (PII redaction, a
> faithfulness judge, MLflow traces) are off by default ([Guardrails and tracing](#guardrails-and-tracing-optional)).
> The judge is not trustworthy yet: it agreed with an AI grader on 34 of 40 answers and caught 1 of 4 unsupported
> answers, and no human has graded either. The tables below were measured on the original PDF revision; AWS has
> since revised it, and only the 4 slow tests were re-run on the new one (they pass).

## Results

**50 questions** about the document: 22 lookups, 8 paraphrases (reworded to avoid the document's key phrases;
some words still overlap), 10 multi-part questions (9 need two separate passages, one is a single sentence)
and 10 unanswerable. The 40 answerable ones are scored for retrieval. A chunk "hits" if it contains the gold
evidence phrase.

**Who wrote them:** the questions, reference answers and evidence phrases were drafted by an AI assistant
(Claude), not by a subject-matter expert, and a script checked that every evidence phrase appears in the PDF.
The first 27 were written before any retrieval was run; the other 23 (paraphrases, multi-part, five more
unanswerable) were added later, after seeing where keyword search failed, so the set is not blind.

### Retrieval (k = 4, 40 answerable questions)

| Retrieval mode | Recall@4 | Complete recall@4 | MRR@4 | Recall@1 | Recall@8 | Missed at k=4 |
|---|---|---|---|---|---|---|
| legacy dense (original notebook index, naive 1000/200 page chunks) | 0.800 | n/a | 0.671 | 0.550 | 0.925 | n/a |
| dense (new chunks, MPNet + FAISS) | 0.825 | 0.775 | 0.613 | 0.450 | 0.900 | q01 q09 q10 q16 q22 p07 p08 |
| BM25 | 0.850 | 0.800 | 0.756 | 0.675 | 0.875 | p01 p02 p03 p04 p07 m02 |
| hybrid (BM25 + dense, RRF) | 0.825 | 0.800 | 0.665 | 0.550 | 0.900 | q01 q09 q22 p02 p03 p04 p07 |
| **hybrid + cross-encoder rerank (shipped)** | **0.900** | **0.875** | **0.781** | 0.675 | **0.975** | p02 p06 p08 m02 |

*Complete recall@4*: every evidence phrase of a multi-part question is in the top 4. The legacy row was
measured once with the original `faiss_index/` from the notebooks; that index and the one-off script are not in
the repo, so this row cannot be reproduced from a fresh clone.

By question type (Recall@4 / MRR@4; the best in each column is bold. Note that plain dense search, not the
shipped mode, is best on paraphrases):

| Retrieval mode | lookup (22) | paraphrase (8) | multi-part (10) |
|---|---|---|---|
| legacy dense | 0.77 / 0.68 | 0.75 / 0.50 | 0.90 / 0.78 |
| dense | 0.77 / 0.59 | **0.75 / 0.54** | 1.00 / 0.71 |
| BM25 | **1.00 / 0.91** | 0.38 / 0.16 | 0.90 / 0.90 |
| hybrid | 0.86 / 0.76 | 0.50 / 0.20 | **1.00 / 0.83** |
| hybrid + rerank | **1.00 / 0.91** | 0.62 / 0.44 | 0.90 / 0.78 |

### Agent (answers, citations, abstention)

One run over all 50 questions with `gpt-5-nano` (reasoning effort `low`), the only model the available OpenAI key
could use. The model is not deterministic, so another run can differ.

| Metric | Result |
|---|---|
| Abstention rate on the 10 unanswerable questions | **10 / 10**; 9 chosen by the model, 1 (u03) forced by the step cap |
| False-abstention rate on the 40 answerable questions | **0 / 40** |
| Answer correctness, strict 1/0, **graded by the AI assistant** (no human check yet) | **35 / 40 (87.5%)**: 3 were partly correct (scored 0) and 2 wrong; 38 / 40 (95%) counting partly correct |
| Answer correctness checked by a human | **pending** |
| Answers with at least one verified citation | 40 / 40 |
| Answers that cite a page containing the evidence phrase | 35 / 40 (87.5%) |
| Strict cited-page precision (cited pages vs evidence pages) | 0.63, a lower bound: citing the FAQ page that restates the same fact counts as wrong |
| Answers citing a page that was never retrieved | 1 / 40 (m06) |
| Tokens for the whole run | 70,652 in + 15,208 out (about 1,400 in / 300 out per question) |
| Latency (sequential, single run) | median 3.9 s, p90 4.9 s, max 17.2 s; not a benchmark |

By type, fully correct answers: lookups 22 / 22, paraphrases 6 / 8, multi-part 7 / 10 (3 partly correct).
The two wrong answers (p02, p08) were both retrieval misses; of the 36 questions whose evidence was retrieved
in the top 4, 34 were answered fully correctly, against 1 of 4 for the retrieval misses. Answer quality here
tracks retrieval quality. The review notes for every non-perfect answer are in
`eval/results/assistant_review.csv`.

What is also verified without LLM tokens: with the real retriever and a scripted stand-in for the LLM, the tool
returns passages labelled with their pages, a citation to a page that was never retrieved is flagged, and the
live server maps a provider rate limit to HTTP 503.

## Findings

What the experiments showed, including the parts that did not go as hoped.

1. **The cross-encoder rerank improved retrieval, mostly on lookups, but it is not the whole story.** Recall@4
   went 0.800 (legacy) -> 0.825 (new dense) -> 0.900 (hybrid + rerank), and MRR@4 0.671 -> 0.613 -> 0.781. It
   fixed 6 questions that dense search missed (5 lookups and one paraphrase) and broke 3 that dense search got
   right (all paraphrase or multi-part). The new heading-first chunking and plain dense search were **not
   clearly better** than the original naive chunks. See [question 3](#3-why-a-cross-encoder-and-did-it-help-paraphrases).
2. **BM25 looks great on lookups and falls apart on paraphrases.** Recall@4 is 1.00 on lookups but 0.38 on
   paraphrases. The lookup questions reuse the document's own words, which flatters keyword search; the 8
   paraphrases exposed it. An earlier, lookup-only comparison had BM25 at Recall@4 = 1.00, which would have
   been misleading.
3. **Plain hybrid search depends strongly on RRF's `k`; the shipped pipeline does not.** With the default
   `k = 60`, hybrid scored Recall@4 0.825, no better than dense. With `k = 1` it scored **0.975**, higher than
   the shipped hybrid + rerank (0.900). That `k` was found by sweeping on the same 40 questions, so it is
   optimistic and the default was not changed; the next step is to choose `k` on held-out questions. The
   shipped mode ignores `k` because the reranker reorders the top 20 of only 35 chunks. See
   [question 1](#1-why-rrf-with-k--60).
4. **Prepending the section heading to each chunk made no clear difference** (hybrid + rerank Recall@4 0.900
   either way; MRR@4 0.781 with it, 0.752 without). It stays on, but this is within noise.
5. **A score-based abstention gate is not good enough.** Using the reranker's best score to refuse questions
   gives AUC 0.80. At the best threshold (-0.40) it rejects 80% of unanswerable questions but also **27.5% of
   answerable ones**, and the threshold was picked on the same questions, so even that is optimistic. The gate
   exists (`MIN_RERANK_SCORE`) but is **off by default**; abstention is left to the prompt, which worked in
   the agent run (finding 10).
6. **The agent can loop, and loops cost real money.** One unanswerable question ("what chunk size does the
   guide recommend?") made the model search 7+ times with new queries; another, answerable, question re-sent
   the identical search 4 times in 3 seconds. The history grows with every search, so one question hit Groq's
   per-request limit (7,612 tokens requested vs a 7,000 limit). Fixes in the code: a step cap per question and a repeated-query guard whose
   reply tells the model to stop searching. With `gpt-5-nano` one question (u03) still hit the step cap in the
   full run, although it stopped after two searches when tried alone, so the behaviour varies between runs.
7. **Two different provider limits.** Groq's `on_demand` tier limits input tokens per minute (7,000) *and* total tokens
   per day (200,000). Retrying with a wait fixes the first, never the second, so the eval runner stops on a daily
   limit instead of retrying, and it resumes from a saved file.
8. **Hard cases are about wording, not chunking.** In notebook experiments two questions ("intended audience",
   "external sources") landed in the bottom half under dense search for both chunk sizes tried (20th and 21st of
   36 chunks; 34th and 45th of 63), probably because words like *guide* and *RAG* appear in nearly every chunk of
   this document (not tested directly). Keyword search or a reranker fixed them; chunk size did not.
9. **A silent crash.** FAISS and PyTorch each ship an OpenMP runtime; on macOS arm64 computing embeddings after
   importing FAISS segfaulted (exit 139). In a minimal reproducer 6 of 6 runs crashed and 3 of 3 passed with
   `OMP_NUM_THREADS=1`, so the package sets it on import. Other combinations of the same libraries did not crash,
   so the exact trigger was not identified.

9b. **Chunk overlap can inflate a first-hit metric, but the effect is smaller than first thought.** With the
    notebook-era page-by-page chunking, 500/200 chunks scored Recall@4 0.850 and MRR@4 0.702 against 0.775 and
    0.581 for 500/100, and 19 of 40 answers sat in two chunks at 500/200 (6 of 40 at 500/100). The ranking
    advantage is concentrated in those duplicated answers (Recall@1 gap 0.26 on them, 0.06 on the rest), but a
    small Recall@4 gap remains on the duplicate-free questions, so it is partly an artifact, not wholly. With the
    shipped heading-first chunker the effect almost disappears. See
    [question 4](#4-why-did-500200-chunks-look-better-than-they-were).
10. **Abstention worked, and the prompt did it.** With the score gate off, the agent abstained on 10 of 10
    unanswerable questions and wrongly abstained on 0 of 40 answerable ones. Nine abstentions were the model's
    own decision (the exact sentence "I could not find this in the document."); one was the step cap.
11. **Checking the "unverified citations" found a bug in the citation parser.** Three answers were flagged. Two were
    false alarms: the model wrote `[S4 p.10]` (passage label plus page) and the parser read the 4 as a page. That
    is fixed and tested. The third is real: m06 cited `[p.12]`, a page the search never returned. Flagging
    citations against what was actually retrieved is what caught it.
12. **Cited pages are a weak accuracy measure.** Only 63% of cited pages are the exact pages holding the
    evidence phrase, but many of the others are the FAQ page restating the same fact. Whether a citation really
    supports a sentence needs a human (or a well-checked judge).
13. **The old "no previous conversation" failure did not reproduce.** In the notebook the Groq/Qwen model answered
    q09 with "I don't have any previous conversation context". With `gpt-5-nano` and the new prompt, q09 is
    answered correctly with a verified page citation. It is not known whether the model or the prompt made the
    difference, because Qwen could not be re-run once its daily limit was spent.

### Caveats on these numbers

- The questions and gold phrases were drafted from the document by the same AI assistant that helped build the
  pipeline, so they are friendlier to it than real users' questions would be. There is one document.
- **40 answerable questions: one question is 2.5 points of recall.** Differences of one or two questions
  (for example the prepend-heading ablation) are noise.
- A hit means the chunk contains the evidence phrase. It does not prove the chunk is enough to answer.
- The "multi-part" questions are about *combining passages*. The source is a writing guide, not a troubleshooting
  manual, so there is no troubleshooting content to test.
- The abstention threshold was calibrated on the evaluation questions (no held-out set).
- The agent results are **one run of one model**, so they carry run-to-run noise (u03 behaved differently in a
  spot test and in the full run).
- The correctness grading was done by the same AI assistant that drafted the questions and reference answers, so
  shared blind spots are possible. A human has not yet checked it; `eval/results/grading.csv` says who graded
  each row.
- `eval/questions.json` contains short verbatim phrases from the PDF as evidence (about 100, each under 150
  characters). They remain AWS's copyrighted text, are included only as short evaluation evidence, and are not
  covered by this repository's MIT license. Remove them if you must not redistribute even short quotes.

## Understanding the choices

Each answer below gives the short version, the reason, and this project's own numbers, plus what **not** to
claim. The numbers come from `eval/experiments/design_choices.py` and `eval/experiments/overlap_check.py`
(retrieval only, no LLM calls). Two of the questions rest on a premise the data does not support; those answers
say so.

**How retrieval works in one paragraph.** A question goes through two cheap searches over the 35 chunks: dense
(embeddings, meaning-based) and BM25 (keywords). RRF merges the two ranked lists. A cross-encoder then re-scores
the best 20 merged chunks and the top 4 go to the LLM. Dense search is good at meaning but blurry; BM25 is exact
but literal; the cross-encoder is accurate but slow, so it only sees the shortlist.

### 1. Why RRF with k = 60?

**The problem it solves.** Dense search gives cosine scores (roughly 0 to 1); BM25 gives unbounded scores (here
up to about 22). You cannot add them. Reciprocal Rank Fusion ignores the scores and uses only the *rank* in each
list: `score(chunk) = sum over lists of 1 / (k + rank)`. A chunk absent from a list adds nothing for it.

**What `k` does.** It sets how much being near the top matters compared with appearing in both lists. Take chunk
A (rank 1 in one list only) and chunk B (rank 3 in both lists):

| k | A | B | winner |
|---|---|---|---|
| 0 | 1.000 | 0.667 | A: the single best rank wins |
| 1 | 0.500 | 0.500 | tie |
| 10 | 0.091 | 0.154 | B |
| 60 | 0.016 | 0.032 | B: agreement between the lists wins |
| 100 | 0.010 | 0.019 | B |

Small `k` rewards "somebody ranked this first"; large `k` rewards "both lists like this one".

**Why 60.** It is the constant from the paper that introduced RRF (Cormack, Clarke and Buttcher, 2009), where it
was chosen empirically, and it is a common default in search engines. **In this project it was a default, not a
tuned value.** A sweep on the 40 questions shows plain hybrid search is sensitive to it (candidate pool 20):

| `k` | hybrid Recall@4 | hybrid MRR@4 | hybrid + rerank Recall@4 / MRR@4 |
|---|---|---|---|
| 1 | **0.975** | 0.752 | 0.900 / 0.781 |
| 10 | 0.850 | 0.673 | 0.900 / 0.781 |
| 60 | 0.825 | 0.665 | 0.900 / 0.781 |
| 100 | 0.825 | 0.665 | 0.900 / 0.781 |

Two things follow. (1) The shipped mode does not care about `k`, because the cross-encoder reorders the top 20,
which is 57% of all 35 chunks, so it sees most of the corpus whatever the fusion did (with a pool of 10 it drops
to 0.875 / 0.769). (2) A small `k` suits this data, probably because BM25 puts exact matches at rank 1 on the
lookup questions and a small `k` lets that single vote win (a guess, not tested directly).

**What not to claim:** "60 is optimal". Say: "60 is the standard default; I checked sensitivity and plain hybrid
prefers a small `k` here, but that was tuned on my test questions, so the right move is to choose it on held-out
data. It does not affect the shipped reranked mode."

### 2. What does BM25's IDF term do?

BM25 scores a chunk against a query like this:

```
score(query, chunk) = sum over query words t of   IDF(t) * tf * (k1 + 1) / ( tf + k1 * (1 - b + b * len / avglen) )
IDF(t) = ln( 1 + (N - n + 0.5) / (n + 0.5) )          N = chunks in the corpus, n = chunks containing t
```

Three ideas: **IDF** weights a word by how rare it is; **tf saturation** (`k1` = 1.5) means repeating a word
helps less each time; **length normalisation** (`b` = 0.75) stops long chunks winning just by being long.

**The IDF term** makes common words nearly worthless and rare words decisive. In this corpus (N = 35):

| word | chunks containing it | IDF |
|---|---|---|
| rag | 34 | 0.043 |
| document | 22 | 0.470 |
| llm | 20 | 0.563 |
| retrieval | 12 | 1.058 |
| vector | 8 | 1.443 |
| table | 4 | 2.079 |
| hallucination | 2 | 2.667 |
| audience, kendra, sequential, numbered, lsh | 1 | 3.178 |

A match on "audience" is worth about 74 matches on "rag". That is why BM25 ranked the "intended audience" and
"external sources" questions first, while dense search, pulled towards the many chunks about RAG in general,
ranked them in the bottom half. (Prepending section headings to chunks adds "rag" to even more chunks, which
lowers its IDF further.)

**Its limit.** IDF only helps when the query shares a rare word with the chunk. "Grids of data" versus "tables"
shares nothing, which is why BM25 scores 0.38 Recall@4 on paraphrases.

### 3. Why a cross-encoder, and did it help paraphrases?

**First, the premise is not what the data shows.** On the 8 paraphrases, plain dense search hit 6 and hybrid +
rerank hit 5. The reranker's real gains were on lookups:

| | dense | hybrid + rerank |
|---|---|---|
| lookups (22) | 17 hits | **22 hits** |
| paraphrases (8) | **6 hits** | 5 hits |
| fixed by the reranker | | q01 q09 q10 q16 q22 p07 |
| broken by the reranker | | p02 p06 m02 |

It did lift paraphrases above BM25 (3 of 8) and plain RRF hybrid (4 of 8), because it cleaned up the fused list.

**How the two models differ.**
- A **dense bi-encoder** turns the query into one vector and each chunk into one vector, *separately*, and compares
  them with a dot product. The chunk vector is computed once, before any question exists, so it must summarise
  everything in the chunk. A 1,000-character chunk about several things becomes a blur, and in a document where
  every chunk is about RAG that blur sits near generic questions. It is fast because chunk vectors are precomputed.
- A **cross-encoder** reads the query and the chunk *together* in one pass, so every query word can attend to
  every chunk word. It can see that "audience" lines up with "intended for AI engineers, data scientists..."
  inside a long chunk. It was trained on relevance judgements (MS MARCO) to output a relevance score. The price:
  one model run per (query, chunk) pair, nothing precomputed, so it is only affordable on a shortlist. That is the
  retrieve-then-rerank pattern.

**Why it did not fix paraphrases.** The four questions the shipped pipeline still misses all had the evidence
chunk inside the 20 candidates; the reranker simply ranked it below the top 4:

| question | evidence chunk's rerank position | its score | 4th place's score |
|---|---|---|---|
| p02 "grids of data" (means tables) | 7th | -8.70 | -8.36 |
| p06 | 5th | -4.07 | -3.93 |
| p08 | 9th | -10.85 | -10.13 |
| m02 | 5th | -7.98 | -7.30 |

In p02, p08 and m02 every candidate scored below -5, meaning the model judged nothing relevant. The model is
small (MiniLM) and trained on web search queries, so it does not bridge "grids of data" to "tables" any better
than the embedding model did. A reranker can only reorder what the first stage found, and it is only as good at
meaning as its training. Fixes listed in the future scope: query rewriting, or a larger reranker.

**What not to claim:** "cross-encoders solve vocabulary mismatch". Say: "They judge query and passage together, so
they are more accurate than bi-encoders but too slow to run on everything. Here it fixed five of the lookup
misses and made paraphrases slightly worse."

### 4. Why did 500/200 chunks look better than they were?

**How the metric counts.** A question is a hit if a retrieved chunk contains the gold phrase; the rank is the
position of the *first* such chunk. With 500-character chunks and 200 overlap, 40% of each chunk repeats the
previous one, so a phrase near a boundary lands in two chunks. That gives it two chances to rank well.

**What was measured.** Rebuilding the notebook-era chunking (page by page, no headings), all 40 questions, dense
search:

| chunking | chunks | answers in 2+ chunks | Recall@4 | MRR@4 | Recall@1 |
|---|---|---|---|---|---|
| 500/200 | 76 | **19 / 40** | 0.850 | 0.702 | 0.600 |
| 500/100 | 63 | 6 / 40 | 0.775 | 0.581 | 0.425 |

Splitting by whether the answer was duplicated:

| questions | 500/200 Recall@4 / MRR@4 / Recall@1 | 500/100 Recall@4 / MRR@4 / Recall@1 |
|---|---|---|
| answer duplicated (23) | 0.870 / 0.790 / 0.739 | 0.826 / 0.638 / 0.478 |
| answer in one chunk (17) | 0.824 / 0.583 / 0.412 | 0.706 / 0.505 / 0.353 |

Most of the *ranking* advantage (Recall@1: 0.26 apart on duplicated answers, 0.06 on the rest) comes from the
duplicated answers, so it was largely an artifact. But a Recall@4 gap remains on the 17 clean questions (two
questions), so it was **partly**, not wholly, an artifact. It is also bad for the product: duplicate chunks fill
the 4 slots the LLM reads with the same sentence.

**With the shipped chunker the effect nearly vanishes**, because sections are split first and overlap only
applies inside a section:

| shipped chunker | chunks | answers in 2+ chunks | dense Recall@4 / MRR@4 | hybrid + rerank Recall@4 / MRR@4 |
|---|---|---|---|---|
| 1000/200 (chosen) | 35 | 3 / 40 | 0.825 / 0.613 | 0.900 / 0.781 |
| 500/200 | 62 | 4 / 40 | 0.850 / 0.696 | 0.875 / 0.787 |
| 500/100 | 60 | 2 / 40 | 0.850 / 0.688 | 0.900 / 0.781 |

That is also why the project stayed at 1000/200: no evidence that smaller chunks help the shipped pipeline.

**What not to claim:** "overlap hurts" or "overlap is all artifact". Say: "A first-hit metric rewards redundancy,
so I counted how many answers sit in more than one chunk before comparing chunk settings."

### 5. What was the abstention gate's AUC?

**0.80** (0.802). The gate takes the reranker's best score for a question and refuses to answer if it is below a
threshold. **AUC** is the chance that a randomly chosen answerable question has a higher best score than a
randomly chosen unanswerable one: 0.5 is a coin flip, 1.0 is perfect separation. 0.80 means the two groups are
ordered correctly in 80% of pairs but overlap heavily.

**What that costs.** The best threshold (-0.40) keeps 29 of 40 answerable questions and rejects 8 of 10
unanswerable ones (balanced accuracy 0.76). So it wrongly refuses **11 of 40 answerable questions (27.5%)** and
lets 2 of 10 unanswerable ones through.

**Why it is weak.** The scores are raw model outputs, not probabilities. They measure how *on topic* a passage
is, not whether it *contains the answer*. The two highest-scoring unanswerable questions show it: u04 ("give the
step-by-step Terraform commands to deploy a RAG application on AWS", score 5.35) shares words with a resource
title in the document ("Deploy a RAG use case on AWS by using Terraform and Amazon Bedrock"), and u05 ("which
embedding model is the best", 1.69) is about topics the guide discusses without recommending a model. Neither
is answerable, but both look relevant. Meanwhile 10 of the 40 answerable questions scored below -3.9 (hard
paraphrases and multi-part questions). The gate does get some cases right: the chunk-size question (u03) scored a
low -9.12.

The threshold was picked on the same questions it is scored on, so even 0.80 is optimistic. The gate is **off
by default**; the prompt did better in the agent run (10 of 10 unanswerable questions abstained, 0 of 40 answerable
ones did).

### Other choices at a glance

| choice | reason | tested? |
|---|---|---|
| Split on headings first, then 1000/200 | keeps a topic together; gives every chunk a page and section for citations | compared with the old chunks: no clear retrieval gain; the benefit is the citations |
| Prepend the heading to the embedded text | tells a short chunk what it is about | ablation: no clear difference (0.900 either way) |
| `all-mpnet-base-v2` embeddings, FAISS flat inner product | strong general-purpose sentence model; a flat index is exact search, and approximate search is pointless at 35 vectors; normalised vectors make inner product equal to cosine | no other embedding model compared |
| Own BM25, `k1` = 1.5, `b` = 0.75 | textbook defaults, no extra dependency | defaults, not tuned |
| 20 candidates, then top 4 | the LLM reads 4 (cost, and fewer distractions); 20 gives the reranker room | 10 candidates: Recall@4 0.875 vs 0.900 |
| Recall@4 and MRR@4 | "does the answer reach the LLM" and "how high is it"; 4 is what the LLM sees | by definition |
| A hit means the chunk contains the evidence phrase | cheap and automatic | known gap: containing the phrase does not prove it is enough to answer |
| Agent with a search tool, not "always retrieve" | the model chooses the query and can search twice | costs one extra LLM call per question |
| Step cap of 6, repeated-query guard | real loops were observed | u03 still hit the cap in one run |
| `gpt-5-nano`, reasoning effort `low` | the only model the available key could use | no other model compared on OpenAI |
| `OMP_NUM_THREADS=1` | stops a segfault between FAISS and PyTorch | latency impact not measured |

## How it works

```
document.pdf
   |  parse.py   PyMuPDF: font size -> headings; drop running headers/footers and the table of
   |             contents; tables become flat "column: value" lines (their own chunks)
   v
 blocks (page, heading level, text)
   |  chunk.py   split on headings first, then 1000 chars / 200 overlap inside each section;
   |             every chunk keeps its page(s) and heading path
   v
 chunks --> index.py: MPNet embeddings in FAISS (saved to data/index, rebuilt if the PDF or settings change)
   |
   |  retrieve.py   dense + BM25 (own implementation, bm25.py) -> RRF fusion -> top 20
   |                -> cross-encoder rerank (ms-marco-MiniLM) -> top 4
   v
 agent.py  LangGraph: LLM -> search_document tool -> LLM, max 6 steps
           answers only from the passages, cites [p.N], says "I could not find this in the document."
           when unsure; citations are checked against what was actually retrieved
   v
 api.py    POST /ask -> answer, abstained, citations, queries, status
```

Page numbers are the **physical PDF page** (page 1 = cover), not the printed footer number.

## Setup

```bash
uv sync --all-extras            # or: pip install -e ".[dev]"
cp .env.example .env            # then put your key in .env (it is gitignored)
```

The LLM is OpenAI (`OPENAI_API_KEY`, default model `gpt-5-nano`, `OPENAI_REASONING_EFFORT=low`) or Groq
(`GROQ_API_KEY`). `LLM_PROVIDER` picks one explicitly; otherwise OpenAI is used whenever its key is set.
`gpt-5-nano` is a reasoning model, so it takes no temperature setting and costs hidden reasoning tokens;
`minimal` effort is cheapest and `low` was used for the results above.

Get the PDF: run `bash scripts/fetch_document.sh` (it downloads *"Writing best practices to optimize RAG
applications"* from [AWS Prescriptive Guidance](https://docs.aws.amazon.com/pdfs/prescriptive-guidance/latest/writing-best-practices-rag/writing-best-practices-rag.pdf)
to `document.pdf`; the file is gitignored because AWS owns the copyright, and the repo never redistributes it).
Any PDF works for the service itself (set `DOCUMENT_PATH`), but page numbers, chunks and every result below depend
on the exact file. The copy used here has SHA-256
`064e6abca87ddeec3d65f000d66ff42c55d6a4c232a7257fa4d6721479072bd6` (check with `shasum -a 256 document.pdf`);
AWS has since revised the PDF; the four slow tests still pass on the newer revision, but the tables below are from the pinned copy.

```bash
python -m rag_app.evaluate retrieval     # builds the index on first use, prints the retrieval table
uvicorn rag_app.api:app --port 8000
curl -s localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question": "Why should numbered lists be sequential?"}'
```

Response shape (illustrative values):

```json
{"answer": "Numbered lists must be sequential, without skipping numbers [p.12].",
 "abstained": false,
 "citations": [{"page": 12, "section": "Documentation best practices for RAG applications",
                "chunk_id": "c020", "quote": "• Ensure numbering is sequential – When using numbered lists..."}],
 "unverified_citation_pages": [], "queries": ["numbered lists sequential"], "status": "ok", "latency_ms": 1800}
```

`unverified_citation_pages` lists pages the model cited that the search never returned (possible hallucinated
citations). The API returns 503 when the LLM provider is rate limited and 502 for other upstream errors.

## Tests

```bash
pytest -m "not slow"     # 33 fast tests, ~1 s: BM25, parser/chunker (synthetic PDFs, incl. a real table), agent, API, LLM provider
pytest                   # + 4 slow tests (need document.pdf, download two models, ~25 s)
```

The slow tests include the regression guard, `tests/test_retrieval_baseline.py`: it fails if Recall@4 or MRR@4
of the shipped pipeline falls below `eval/baseline.json`, or if it becomes worse than plain dense search. It was
checked to fail when the baseline is raised above the current score. To accept a new, better baseline
after an improvement: `python -m rag_app.evaluate retrieval --write-baseline`.

## Evaluating the agent

Needs LLM tokens: the full 50-question run above used about 86,000 tokens (70.6k in, 15.2k out) and took about
12 minutes, one question at a time.

```bash
python -m rag_app.evaluate agent --pause 0     # resumable: appends to eval/results/agent_runs.jsonl
python -m rag_app.evaluate grade               # first call writes eval/results/grading.csv
# open grading.csv and fill the `correct` column (1 or 0) for the answerable rows by hand, then:
python -m rag_app.evaluate grade               # correctness, abstention rate, citation checks
```

**Grading rubric.** `correct = 1` only if the answer states the key points of the reference answer without a
material error or a missing key part; extra detail is fine if the document supports it. A partly correct answer
scores 0 and the notes column says so. Unanswerable rows are graded automatically (correct = the agent abstained).
Every row has a `graded_by` column. The current 40 answerable rows say "assistant, at the owner's request (no
human check yet)"; to record your own grading, change `correct` and set `graded_by` to your name.
`eval/results/assistant_review.csv` keeps the three-level verdicts (1, 0.5, 0) behind those scores.

Citations are re-checked from the saved answer text and retrieved pages every time you grade, so a fix to the
citation parser applies to old runs. The stored runs have the PDF excerpts removed from the citations.

## Docker

```bash
docker build -t doc-rag .
docker run --rm -p 8000:8000 \
  -v "$PWD/.env:/app/.env:ro" -v "$PWD/document.pdf:/app/document.pdf:ro" doc-rag
```

Mount `.env` as a file instead of using `--env-file`: Docker's `--env-file` rejects lines such as
`GROQ_API_KEY = ...` (spaces around `=`), while the app's own loader accepts them. The image does not contain
`.env` or the PDF; both are supplied at run time.

**Built and checked on 2026-10-04** (Apple-silicon Mac, Docker Desktop 29.2, `linux/arm64`):
- The build succeeded on the first try. The image is 3.13 GB, with CPU-only PyTorch (`2.14.1+cpu`) and both
  models baked in (507 MB); it holds no `.env`, no PDF and no key-like environment variables.
- A container started from it was ready in 21.5 s, including building the index from the mounted PDF, reported
  `healthy`, and used about 545 MB of memory.
- `POST /ask` answered a real question with a verified `[p.12]` citation, abstained on "What is the refund
  policy?", and returned 422 for a too-short question.
- `python -m rag_app.evaluate retrieval` inside the Linux container printed a table **identical** to the one
  from macOS on all 13 lines, so the results do not depend on the operating system or the PyTorch build.

The `linux/amd64` build is checked in CI (GitHub's Ubuntu runner builds the image, starts it and gets a 200 from
`/health`); that check does not run the retrieval evaluation inside the container. Not checked: a cold start with no
network, and behaviour under concurrent requests.

## Guardrails and tracing (optional)

All three are off by default, so the base service, the Docker image and the numbers above are unchanged. Install
with `pip install -e ".[guardrails]"` and `python -m spacy download en_core_web_md`, then set the flags in `.env`.

| Flag | What it does |
|---|---|
| `PII_REDACTION=1` | Presidio replaces card numbers (Luhn-checked), emails, phones, IPs, IBANs, SSNs and full names in the question with tags like `<CREDIT_CARD>` **before** the LLM, the logs or the traces see it. The response carries `pii_redacted`. |
| `FAITHFULNESS_CHECK=1` | A second model (`JUDGE_PROVIDER` / `JUDGE_MODEL`, default Groq `openai/gpt-oss-120b`) splits the answer into claims and checks each against the retrieved passages. The response carries `faithfulness` (0-1) and `unsupported_claims`. `MIN_FAITHFULNESS` turns the score into an abstention (`status: low_faithfulness`); leave it unset until you have calibrated it. |
| `MLFLOW_TRACKING_URI=sqlite:///mlflow.db` | One MLflow trace per `/ask` with every LLM and tool span, tagged with abstained, faithfulness, unverified citations, PII types, latency and tokens. View it with `mlflow server --backend-store-uri sqlite:///mlflow.db`. `python -m rag_app.monitor` prints request count, abstain rate, mean faithfulness, p50/p95 latency, tokens and PII counts. |

**PII measured** (`eval/pii_questions.json`: 20 questions with planted cards, emails, phones, names, an SSN, an IP and an
IBAN): all 20 planted values were removed from the text, 19 of 20 with the right entity type (one US SSN was tagged
`PHONE_NUMBER`). **0 of the 50 eval questions were changed**, so no false positives. Two choices came from
measuring: the score threshold is 0.4 because phone numbers score below 0.6 and were missed at 0.6, and a name is
redacted only when it is two or more words, because spaCy tagged "Terraform" as a person at the same score as real
names. The cost: a lone first name is not redacted.

**Faithfulness judge, validated** (`eval/results/faithfulness_calibration.csv`; judge = Groq `openai/gpt-oss-120b`,
answerer = `gpt-5-nano`, a different model). The 40 stored answerable answers were graded claim by claim against the
passages they were written from, **by Claude, not by a human**, so treat the grades as a second opinion and
re-grade them yourself before relying on the number.
- The judge and the grader agreed on **34 / 40 (85%)** answers being fully supported or not.
- The grader found 4 answers with an unsupported claim (p02, p08, m02, m06). The judge caught 1 of them (m06,
  which names the wrong "first step"); it missed the other 3. It also flagged 3 answers (q03, q08, m01) that the
  grader considered supported, borderline paraphrases. So its scores are not reliable enough to block answers,
  and `MIN_FAITHFULNESS` stays unset. 4 bad answers is a small sample.
- A planted check works: with the correct passage, an answer scored 1.0; adding one invented sentence ("lists must
  never exceed ten items") dropped it to 0.5 and named that claim.

**Monitoring snapshot** (2026-10-05, `python -m rag_app.monitor`, after running the 50 eval and 20 PII questions
through `/ask` with all three guardrails on, `eval/experiments/run_traffic.py`; `gpt-5-nano` answers):

| Metric | Value |
|---|---|
| Requests | 70 |
| Abstain rate | 25.7% (10 of the 50 eval questions are unanswerable by design) |
| Mean faithfulness (answered requests) | 0.995 |
| Requests with an unverified citation | 2.9% |
| Latency p50 / p95 (includes the judge call) | 5.97 s / 8.86 s |
| Mean input / output tokens | 1374 / 355 |
| Requests with PII redacted | 20 (email 6, phone 6, person 5, card 5, IBAN 1, IP 1) |

The judge is the weak link: a mean of 0.995 mostly shows it rarely flags anything, which the calibration above
says is not the same as answers being faithful.

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request, with no API keys or secrets (the tests use stub
retrievers and a scripted chat model, so CI never calls an LLM):
1. **test**: installs CPU-only PyTorch, runs the 33 fast tests, downloads the PDF with `scripts/fetch_document.sh`,
   then runs the 4 slow tests, including the Recall@4 / MRR@4 regression guard against `eval/baseline.json`.
2. **docker**: builds the image, starts it and checks `GET /health`; on pushes to `main` it pushes the image to
   GitHub Container Registry using the built-in `GITHUB_TOKEN`.

Secrets live only in `.env` (gitignored; `.env.example` has placeholders). Never commit keys: pass them to
containers at run time.

## Publishing to GitHub

This folder is its own git repository (created with `git init` inside the project, because the enclosing
repository on this machine is the **home directory**; never run `git add` from there). It is pushed to a
**public** GitHub repository, `Sidhartht1607/doc-rag`, on branch `main` (created and pushed with the `gh` CLI;
GitHub holds 40 files and none of `.env`, the PDF or the indexes). It was made public on 2026-10-06, after the
licence points below were reviewed.

Points that were settled before making it public:
- the PDF is AWS copyrighted material and is not in the repo, but `eval/questions.json` holds short verbatim
  phrases from it as evidence;
- the questions and reference answers were drafted by an AI assistant, and the grading is the assistant's
  (see `graded_by` in `eval/results/grading.csv`);
- ~~the repo has no `LICENSE` file yet~~: done, MIT (see `LICENSE`).

The original exploration files at the project root are gitignored; the committed copies live in `notebooks/` with
their outputs cleared, because the outputs contain the PDF's text.

## Project layout

```
rag_app/      parse.py chunk.py index.py bm25.py retrieve.py agent.py api.py evaluate.py config.py guardrails.py monitor.py
eval/         questions.json (50 questions)  baseline.json  pii_questions.json (20 planted-PII questions)
              results/ retrieval_results.{json,md}  agent_runs.jsonl  grading.csv  assistant_review.csv
eval/experiments/  design_choices.py  overlap_check.py   (reproduce the numbers in "Understanding the choices")
notebooks/    the exploration notebooks, outputs cleared, with the legacy rag.py / eval_data.py they import
tests/        fast unit tests + slow regression tests
Dockerfile  requirements.txt (locked, exported)  pyproject.toml  uv.lock
```

The exploration that led here is in `notebooks/` (outputs cleared; run them from a folder that also has
`document.pdf`, and install with `uv sync --extra notebooks`). The working originals stay at the project root and
are not committed.

## Future scope

### Next: production checks (not done yet)

The evaluation harness is in place: 50 questions, retrieval metrics, a regression test, and a resumable agent
run with a hand-grading step, and one complete agent run. What the service still lacks is everything that makes it safe and predictable to
run for other people. None of the four items below exists yet, and none has been measured.

**1. Caching** (today: only the loaded models and index are reused; every question repeats embedding,
search, rerank and the LLM calls)
- Cache the query embedding and the final answer for repeated questions. Key the cache on the normalized
  question *plus* the index fingerprint, model names and prompt version, so an index rebuild or a prompt change
  can never serve a stale answer.
- Measure the hit rate on realistic traffic before trusting it. Be careful with semantic (similarity-based)
  caches: a near-match can return the wrong answer, so they need their own evaluation.
- Check what prompt caching, if any, the LLM provider offers for the fixed system prompt.

**2. Authentication and abuse limits** (today: `/ask` is open to anyone who can reach the port)
- API keys or bearer tokens on `/ask` (leave `/health` open), keys loaded from the environment and never logged.
- Per-key rate limiting and daily token budgets, because one caller can currently burn the whole provider quota
  (this is exactly what happened to the Groq free tier during development).
- Request-size and timeout limits, CORS settings, and tests for 401, 403 and 429 responses.

**3. Tracing and observability** (today: with `MLFLOW_TRACKING_URI` set, every `/ask` writes one MLflow trace with
the LLM and tool spans and tags for abstention, faithfulness, unverified citations, PII types, latency and tokens,
and `python -m rag_app.monitor` summarises them; see [Guardrails and tracing](#guardrails-and-tracing-optional).
Tracing is off by default.)
- Still missing: spans for the stages inside a search (query embedding, FAISS, BM25, fusion, rerank), which the
  LangChain auto-logging does not see.
- Estimated cost per request, a request id that appears in the logs and the response, and structured logs for
  step-limit hits and rate-limit errors.
- Alerting on the signals the experiments showed matter (abstention share, unverified citations, step-cap hits):
  `monitor.py` reports them on demand, but nothing watches or alerts.

**4. Latency benchmarks** (today: never measured; the only timing so far is that the 160 searches of the
retrieval evaluation took about 23 seconds, which is not a latency figure)
- Report p50, p95 and p99 per stage and end to end, separating cold start (model loading and index build) from
  warm requests.
- Load-test `/ask` at increasing concurrency (for example with k6 or Locust) and note where throughput stops
  scaling or the provider starts returning 429s.
- Re-check two settings that were chosen for correctness, not speed: `OMP_NUM_THREADS=1` (added to avoid a
  crash, and it may slow embedding and reranking) and the 20 rerank candidates (try 10 and compare quality and
  latency together).
- Add a latency budget to the test suite the way Recall@4 already has a baseline, so a slow change fails
  loudly.

### Measurements still open
- **Have a human check the grading** in `eval/results/grading.csv`, starting with the five answers scored 0 and
  a random sample of the 35 scored 1; every disagreement is worth reading.
- **Choose RRF's `k` on held-out questions** (the sweep on the evaluation questions favoured `k = 1`).
- Repeat the agent run several times to measure run-to-run variance, and run a second model (Groq/Qwen once its
  limit resets) so the comparison is not one model's behaviour.
- Replace the questions drafted by the assistant with a set other people wrote, grow it past 100, report
  confidence intervals, and keep a held-out split for the abstention threshold.

### Manual steps that could be automated
- **Grading:** a judge now exists (`FAITHFULNESS_CHECK=1`, Groq `openai/gpt-oss-120b`, a different model from the
  answerer) but it is not good enough to replace a human: it agreed with an AI grader on 34 of 40 answers and caught
  1 of 4 answers with an unsupported claim. A human still has to grade `grading.csv`, and the judge's own
  calibration was graded by an AI.
- **README tables** are copied by hand from the files in `eval/results/`; generate them from those files so the
  numbers cannot drift.
- **Re-running the evaluations** after each change (retrieval is free and takes seconds; the agent run costs
  about 86k tokens): a scheduled or per-change job with a token budget.
- **Updating the baseline** (`--write-baseline`) is a manual decision today; require it to appear in the pull
  request that improves quality.
- **Clearing notebook outputs** before committing: a pre-commit hook instead of remembering.
- **Fetching the PDF:** done for CI by `scripts/fetch_document.sh`, which downloads it from AWS at run time without
  storing it. It only warns when AWS revises the file; a pinned, verified copy would need a place to host it that
  the licence allows.

### Retrieval and agent quality
- The paraphrase misses (p02, p06, p08) point to query rewriting or multi-query/HyDE, a larger reranker, or an
  embedding model fine-tuned on question/passage pairs.
- Tune the fusion (weights instead of plain RRF), try parent-document retrieval, and use layout-aware parsing for
  documents with real tables (the table path is tested on a synthetic PDF, but this document has no real table).
- Make abstention reliable with a separate answerability check or verifier step instead of one prompt sentence.
- Structured output (claims with citation spans), highlighted source passages with page images, streaming, and
  conversation memory.

### Product and delivery
**Done:** CI runs on every push and pull request (see [Continuous integration](#continuous-integration)): the 33 fast tests,
the 4 slow tests including the Recall@4 / MRR@4 regression guard (with the embedding and reranker models cached
between runs), and a Docker build on Ubuntu (`amd64`) that starts the container and checks `/health`. On pushes to
`main` the image is pushed to GitHub Container Registry as `ghcr.io/sidhartht1607/doc-rag`. A pull request that
raises the Recall@4 baseline above what retrieval achieves fails the guard (PR #1, closed unmerged on purpose).
The package is public (GHCR makes new packages private even for a public repo, so this was set by hand in the
package settings); anonymous access to the tag list and the `latest` manifest was checked on 2026-10-06 through the
registry API, without a logged-in session. A full `docker pull` from a logged-out machine has not been run.

**Still open:**
- Put the optional guardrails in the image: Presidio and its spaCy model are not installed in the `Dockerfile`
  (roughly 0.5 GB more), so `PII_REDACTION=1` only works outside the container today.
- A persistent index volume (the index is rebuilt on first start of a fresh container unless `/app/data/index` is
  mounted).
- Several documents (per-document filters and metadata).
- Swap the source for a document that can be redistributed. CI downloads the AWS PDF at run time and the repo
  never stores it, but the tests and results still depend on a file the repo does not own.
