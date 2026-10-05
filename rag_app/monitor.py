"""Summarise recent /ask traces from MLflow.   python -m rag_app.monitor [--limit 500]

Needs MLFLOW_TRACKING_URI (and the guardrails extra). Prints request count, abstain rate, mean faithfulness,
unverified-citation share, p50/p95 latency, mean tokens and redacted-PII counts by type.
"""
import argparse
import os
from collections import Counter

import mlflow


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarise(rows: list[dict]) -> dict:
    """`rows` are trace tag dicts; pure so it can be tested without a server."""
    n = len(rows)
    if not n:
        return {"requests": 0}
    col = lambda k: [v for v in (_num(r.get(k)) for r in rows) if v is not None]
    lat, faith = sorted(col("latency_ms")), col("faithfulness")
    pct = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))] if lat else None
    pii = Counter(t for r in rows for t in (r.get("pii_types") or "").split(",") if t)
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    return {
        "requests": n,
        "abstain_rate": round(sum(r.get("abstained") == "True" for r in rows) / n, 3),
        "mean_faithfulness": mean(faith),
        "unverified_citation_share": round(sum((_num(r.get("unverified_citations")) or 0) > 0 for r in rows) / n, 3),
        "latency_ms_p50": pct(0.5), "latency_ms_p95": pct(0.95),
        "mean_input_tokens": mean(col("input_tokens")), "mean_output_tokens": mean(col("output_tokens")),
        "requests_with_pii": sum(1 for r in rows if r.get("pii_types")),
        "pii_by_type": dict(pii),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "doc-rag"))
    traces = mlflow.search_traces(max_results=args.limit, return_type="list")
    rows = [dict(t.info.tags or {}) for t in traces]
    for key, value in summarise(rows).items():
        print(f"{key:28} {value}")


if __name__ == "__main__":
    main()
