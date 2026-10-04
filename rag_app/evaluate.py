"""Evaluation: retrieval metrics (local, free) and agent metrics (calls the LLM).

    python -m rag_app.evaluate retrieval [--write-baseline]   # Recall@4, MRR@4 for every retrieval mode
    python -m rag_app.evaluate calibrate                       # pick the abstention score threshold
    python -m rag_app.evaluate agent [--limit N]               # run the agent, save answers + citations
    python -m rag_app.evaluate grade                           # score the hand-graded CSV
"""
import argparse
import csv
import json
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from .config import ROOT, get_settings
from .retrieve import MODES, Retriever

QUESTIONS_PATH = ROOT / "eval" / "questions.json"
BASELINE_PATH = ROOT / "eval" / "baseline.json"
RESULTS_DIR = ROOT / "eval" / "results"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).lower()


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    return json.loads(Path(path).read_text())


# ------------------------------------------------------------------------------------------
# retrieval
# ------------------------------------------------------------------------------------------
def evidence_ranks(question: dict, hits) -> list[int | None]:
    """For each evidence phrase: rank of the first retrieved chunk that contains it (None = missed)."""
    texts = [normalize(h.chunk.text) for h in hits]
    return [
        next((rank for rank, t in enumerate(texts, start=1) if normalize(phrase) in t), None)
        for phrase in question["evidence"]
    ]


def retrieval_metrics(retriever: Retriever, questions: list[dict], mode: str, k: int = 4) -> dict:
    """Recall@k (any evidence phrase found), complete recall@k (all found), MRR@k.

    Only questions that have evidence count; the unanswerable ones have nothing to retrieve.
    """
    scored = [q for q in questions if q["evidence"]]
    rows = []
    for q in scored:
        hits = retriever.search(q["question"], k=max(k, 8), mode=mode)   # k=8 gives Recall@8 for free
        ranks = evidence_ranks(q, hits)
        found = [r for r in ranks if r is not None]
        rows.append({"id": q["id"], "type": q["type"], "first": min(found) if found else None,
                     "all": len(found) == len(ranks) and all(r <= k for r in found)})

    def summarise(subset):
        n = len(subset) or 1
        within = lambda r, kk: r is not None and r <= kk
        return {
            "n": len(subset),
            f"recall@{k}": sum(within(r["first"], k) for r in subset) / n,
            f"complete_recall@{k}": sum(r["all"] for r in subset) / n,
            f"mrr@{k}": sum(1 / r["first"] for r in subset if within(r["first"], k)) / n,
            "recall@1": sum(within(r["first"], 1) for r in subset) / n,
            "recall@8": sum(within(r["first"], 8) for r in subset) / n,
        }

    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    return {
        "mode": mode, "k": k,
        "overall": summarise(rows),
        "by_type": {t: summarise(rs) for t, rs in sorted(by_type.items())},
        "misses": [r["id"] for r in rows if not (r["first"] is not None and r["first"] <= k)],
    }


def run_retrieval_eval(retriever: Retriever, questions: list[dict], k: int = 4, modes=MODES) -> dict:
    return {m: retrieval_metrics(retriever, questions, m, k) for m in modes}


def markdown_table(results: dict, k: int = 4) -> str:
    lines = [
        f"| Retrieval mode | Recall@{k} | Complete recall@{k} | MRR@{k} | Recall@1 | Recall@8 | Missed at k={k} |",
        "|---|---|---|---|---|---|---|",
    ]
    for mode, r in results.items():
        o = r["overall"]
        lines.append(
            f"| {mode} | {o[f'recall@{k}']:.3f} | {o[f'complete_recall@{k}']:.3f} | {o[f'mrr@{k}']:.3f} "
            f"| {o['recall@1']:.3f} | {o['recall@8']:.3f} | {', '.join(r['misses']) or '-'} |"
        )
    return "\n".join(lines)


def type_table(results: dict, k: int = 4) -> str:
    types = sorted({t for r in results.values() for t in r["by_type"]})
    lines = ["| Retrieval mode | " + " | ".join(f"{t} (Recall@{k} / MRR@{k})" for t in types) + " |",
             "|---|" + "---|" * len(types)]
    for mode, r in results.items():
        cells = [f"{r['by_type'][t][f'recall@{k}']:.2f} / {r['by_type'][t][f'mrr@{k}']:.2f}" for t in types]
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------
# abstention threshold calibration
# ------------------------------------------------------------------------------------------
def calibrate(retriever: Retriever, questions: list[dict]) -> dict:
    """Best cross-encoder score per question, split by answerable / unanswerable.

    The threshold is picked on these same questions, so treat the result as optimistic.
    """
    best = []
    for q in questions:
        hits = retriever.search(q["question"], k=retriever.settings.top_k, mode="hybrid_rerank")
        best.append((q["id"], bool(q["evidence"]), hits[0].score if hits else float("-inf")))
    pos = [s for _, ans, s in best if ans]          # answerable
    neg = [s for _, ans, s in best if not ans]      # unanswerable
    auc = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
    candidates = sorted({s for _, _, s in best})
    thresholds = [(a + b) / 2 for a, b in zip(candidates, candidates[1:])]

    def balanced(t):
        tpr = sum(s >= t for s in pos) / len(pos)   # answerable kept
        tnr = sum(s < t for s in neg) / len(neg)    # unanswerable rejected
        return (tpr + tnr) / 2, tpr, tnr

    t = max(thresholds, key=lambda x: balanced(x)[0])
    bal, tpr, tnr = balanced(t)
    return {"auc": auc, "threshold": t, "balanced_accuracy": bal, "answerable_kept": tpr,
            "unanswerable_rejected": tnr,
            "answerable_scores": sorted(round(s, 2) for s in pos),
            "unanswerable_scores": sorted(round(s, 2) for s in neg)}


# ------------------------------------------------------------------------------------------
# agent evaluation (calls the LLM; resumable so a rate limit never costs you finished questions)
# ------------------------------------------------------------------------------------------
RUNS_PATH = RESULTS_DIR / "agent_runs.jsonl"
GRADING_PATH = RESULTS_DIR / "grading.csv"


def run_agent_eval(agent, questions: list[dict], out_path: Path = RUNS_PATH, limit: int | None = None,
                   pause: float = 14, retries: int = 3, retry_wait: float = 30) -> None:
    """Ask every question once and append each result to a jsonl file. Already-done ids are skipped."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in out_path.read_text().splitlines()} if out_path.exists() else set()
    todo = [q for q in questions if q["id"] not in done][:limit]
    print(f"{len(done)} already done, running {len(todo)}")
    for q in todo:
        result = None
        for attempt in range(1, retries + 1):
            try:
                result = agent.ask(q["question"]).to_dict()
                break
            except Exception as exc:
                if type(exc).__name__ != "RateLimitError":
                    raise
                if "per day" in str(exc) or "insufficient_quota" in str(exc):
                    print("Stopped: daily token limit or account quota reached.", str(exc)[:200])
                    return
                print(f"  {q['id']}: rate limited (try {attempt}/{retries}), waiting {retry_wait}s")
                time.sleep(retry_wait)
        if result is None:
            print(f"  {q['id']}: gave up after {retries} tries; run again to resume")
            return
        row = {"id": q["id"], "type": q["type"], "question": q["question"], "llm": agent.llm_name, **result}
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  {q['id']}: {'abstained' if result['abstained'] else 'answered'}, {result['llm_calls']} llm calls")
        time.sleep(pause)


def load_runs(path: Path = RUNS_PATH) -> dict[str, dict]:
    """Saved runs, with citations re-checked from the answer text and the pages that were retrieved.

    Recomputing here (instead of trusting the stored fields) means a fix to the citation parser
    applies to old runs without asking the LLM again.
    """
    from .agent import split_citations

    runs = {r["id"]: r for r in map(json.loads, Path(path).read_text().splitlines())}
    for run in runs.values():
        if run["abstained"]:
            run["cited_verified"], run["cited_unverified"] = [], []
        else:
            run["cited_verified"], run["cited_unverified"] = split_citations(
                run["answer"], [h["pages"] for h in run["retrieved"]])
    return runs


def make_grading_csv(questions: list[dict], runs_path: Path = RUNS_PATH, out_path: Path = GRADING_PATH) -> int:
    """One row per answered question. Fill the `correct` column (1 or 0) by hand for answerable rows.

    Unanswerable rows are graded automatically: correct = the agent abstained.
    """
    runs = load_runs(runs_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "type", "question", "reference_answer", "agent_answer", "abstained",
              "cited_pages", "gold_pages", "correct", "notes"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for q in questions:
            run = runs.get(q["id"])
            if run is None:
                continue
            auto = "" if q["evidence"] else str(int(run["abstained"]))
            writer.writerow({
                "id": q["id"], "type": q["type"], "question": q["question"],
                "reference_answer": q["reference_answer"], "agent_answer": run["answer"],
                "abstained": int(run["abstained"]),
                "cited_pages": ",".join(map(str, run["cited_verified"])),
                "gold_pages": ",".join(map(str, q.get("gold_pages", []))),
                "correct": auto, "notes": "",
            })
    return len(runs)


def grade(questions: list[dict], runs_path: Path = RUNS_PATH, grading_path: Path = GRADING_PATH) -> dict:
    runs = load_runs(runs_path)
    by_id = {q["id"]: q for q in questions}
    answerable = [i for i in runs if by_id[i]["evidence"]]
    unanswerable = [i for i in runs if not by_id[i]["evidence"]]
    out = {
        "questions_run": len(runs),
        "abstention_rate_on_unanswerable": (
            sum(runs[i]["abstained"] for i in unanswerable) / len(unanswerable) if unanswerable else None),
        "false_abstention_rate_on_answerable": (
            sum(runs[i]["abstained"] for i in answerable) / len(answerable) if answerable else None),
        # abstentions forced by the step cap rather than chosen by the model
        "abstentions_forced_by_step_cap": sorted(i for i, r in runs.items() if r["status"] == "step_limit"),
    }
    answered = [i for i in answerable if not runs[i]["abstained"]]
    if answered:
        out["answers_with_a_verified_citation"] = sum(bool(runs[i]["cited_verified"]) for i in answered) / len(answered)
        cited = [(pg, by_id[i].get("gold_pages", [])) for i in answered for pg in runs[i]["cited_verified"]]
        # strict: only the pages holding the evidence phrase count, so a correct citation of the FAQ
        # page that restates the same fact is scored wrong. A lower bound, not an accuracy.
        out["cited_page_precision_strict"] = sum(p in gold for p, gold in cited) / len(cited) if cited else None
        out["answers_with_unverified_citation"] = sum(bool(runs[i]["cited_unverified"]) for i in answered) / len(answered)
        out["answers_that_cite_an_evidence_page"] = sum(
            any(p in by_id[i].get("gold_pages", []) for p in runs[i]["cited_verified"]) for i in answered) / len(answered)
    # graded correctness, only for rows where the `correct` column has been filled in (see `graded_by`)
    if Path(grading_path).exists():
        graded = [r for r in csv.DictReader(open(grading_path)) if r["correct"] in ("0", "1") and by_id[r["id"]]["evidence"]]
        out["graded_answerable"] = len(graded)
        out["graded_by"] = dict(sorted(Counter(r.get("graded_by", "unknown") for r in graded).items()))
        if graded:
            out["answer_correctness"] = sum(r["correct"] == "1" for r in graded) / len(graded)
            per_type = defaultdict(list)
            for r in graded:
                per_type[r["type"]].append(r["correct"] == "1")
            out["answer_correctness_by_type"] = {t: sum(v) / len(v) for t, v in sorted(per_type.items())}
    review = Path(grading_path).parent / "assistant_review.csv"
    if review.exists():                      # a review done by the AI assistant, NOT the hand grading
        rows = list(csv.DictReader(open(review)))
        verdicts = [float(r["verdict"]) for r in rows]
        out["assistant_review"] = {
            "n": len(rows),
            "fully_correct": sum(v == 1 for v in verdicts) / len(verdicts),
            "fully_or_partly_correct": sum(v >= 0.5 for v in verdicts) / len(verdicts),
            "wrong": [r["id"] for r in rows if float(r["verdict"]) == 0],
            "partial": [r["id"] for r in rows if float(r["verdict"]) == 0.5],
        }
    return out


# ------------------------------------------------------------------------------------------
# command line
# ------------------------------------------------------------------------------------------
def _retriever() -> Retriever:
    from .index import load_or_build_index

    return Retriever(load_or_build_index(get_settings()))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rag_app.evaluate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retrieval")
    r.add_argument("--k", type=int, default=4)
    r.add_argument("--write-baseline", action="store_true", help="store the default mode's numbers as the test baseline")
    sub.add_parser("calibrate")
    a = sub.add_parser("agent", help="run the agent over the questions (uses the LLM)")
    a.add_argument("--limit", type=int, default=None)
    a.add_argument("--pause", type=float, default=14, help="seconds between questions (provider rate limits)")
    sub.add_parser("grade", help="write grading.csv, then score it once `correct` is filled in")
    args = parser.parse_args(argv)

    questions = load_questions()
    if args.cmd == "retrieval":
        retriever = _retriever()
        results = run_retrieval_eval(retriever, questions, args.k)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "retrieval_results.json").write_text(json.dumps(results, indent=2))
        text = markdown_table(results, args.k) + "\n\n" + type_table(results, args.k)
        (RESULTS_DIR / "retrieval_results.md").write_text(text + "\n")
        print(text)
        if args.write_baseline:
            default = get_settings().default_mode
            o = results[default]["overall"]
            BASELINE_PATH.write_text(json.dumps({
                "mode": default, "k": args.k,
                f"recall@{args.k}": o[f"recall@{args.k}"], f"mrr@{args.k}": o[f"mrr@{args.k}"],
                "dense_recall@4": results["dense"]["overall"]["recall@4"],
                "n_questions": o["n"],
            }, indent=2) + "\n")
            print(f"\nwrote {BASELINE_PATH}")
    elif args.cmd == "calibrate":
        print(json.dumps(calibrate(_retriever(), questions), indent=2))
    elif args.cmd == "agent":
        from .agent import Agent

        run_agent_eval(Agent(_retriever()), questions, limit=args.limit, pause=args.pause)
    elif args.cmd == "grade":
        if not GRADING_PATH.exists():
            n = make_grading_csv(questions)
            print(f"wrote {GRADING_PATH} ({n} rows). Fill the `correct` column (1/0) for answerable rows, then run again.")
        print(json.dumps(grade(questions), indent=2))


if __name__ == "__main__":
    main()
