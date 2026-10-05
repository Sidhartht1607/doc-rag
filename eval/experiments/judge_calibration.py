"""Run the faithfulness judge over the 40 stored answers (needs document.pdf and the judge's API key in .env).

   S=/some/scratch/dir PYTHONPATH=. python eval/experiments/judge_calibration.py

Writes $S/judged.json (answer, passages, judge score and per-claim verdicts). The grades in
eval/results/faithfulness_calibration.csv were made by reading each answer against those passages.
"""
import ast, json, os, sys
from dataclasses import replace
from dotenv import load_dotenv
load_dotenv(".env")
from rag_app.config import get_settings
from rag_app.index import build_index
from rag_app.agent import Agent
from rag_app.guardrails import judge_faithfulness
s = replace(get_settings(), index_dir=__import__("pathlib").Path(os.environ["S"]) / "idx_pinned")
idx = build_index(s)
chunks = {c.chunk_id: c for c in idx.chunks}
runs = [json.loads(l) for l in open("eval/results/agent_runs.jsonl")]
agent = Agent(None, settings=s)
out = []
for r in runs:
    if r["abstained"] in (True, "True"):
        continue
    retr = ast.literal_eval(r["retrieved"]) if isinstance(r["retrieved"], str) else r["retrieved"]
    ids = list(dict.fromkeys(h["chunk_id"] for h in retr))
    passages = [chunks[i].text for i in ids]
    import time
    for attempt in range(8):
        try:
            score, claims = judge_faithfulness(agent.judge, passages, r["answer"]); break
        except Exception as e:
            if "RateLimit" not in type(e).__name__ or attempt == 7: raise
            time.sleep(20)
    time.sleep(8)
    out.append({"id": r["id"], "question": r["question"], "answer": r["answer"], "passages": [f"[p.{chunks[i].page}] {chunks[i].text}" for i in ids], "judge_score": score, "judge_claims": claims})
    print(r["id"], round(score, 2), flush=True)
json.dump(out, open(os.environ["S"] + "/judged.json", "w"), indent=1)
