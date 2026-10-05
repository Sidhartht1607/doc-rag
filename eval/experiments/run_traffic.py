"""Send the 50 eval questions and the 20 planted-PII questions through /ask with every guardrail and tracing on.

   MLFLOW_TRACKING_URI=sqlite:///mlflow.db PYTHONPATH=. python eval/experiments/run_traffic.py
   MLFLOW_TRACKING_URI=sqlite:///mlflow.db python -m rag_app.monitor
"""
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PII_REDACTION", "1")
os.environ.setdefault("FAITHFULNESS_CHECK", "1")
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

from fastapi.testclient import TestClient  # noqa: E402

from rag_app.api import app  # noqa: E402

root = Path(__file__).resolve().parent.parent
questions = [q["question"] for q in json.loads((root / "questions.json").read_text())]
questions += [q["question"] for q in json.loads((root / "pii_questions.json").read_text())]
with TestClient(app) as client:
    for i, q in enumerate(questions, 1):
        for attempt in range(4):                       # the judge's free tier allows ~8k tokens a minute
            r = client.post("/ask", json={"question": q})
            if r.status_code != 503 and r.status_code != 502:
                break
            time.sleep(20)
        body = r.json() if r.status_code == 200 else {}
        print(i, r.status_code, body.get("status"), body.get("faithfulness"), body.get("pii_redacted"), flush=True)
        time.sleep(6)
