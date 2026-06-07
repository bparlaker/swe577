"""End-to-end RAG experiment.

Runs every question in ``data/questions.json`` through two configurations:

1. **Bare** - the chosen backend answers from its parameters alone, with
   no retrieval.
2. **KG-grounded (RAG)** - the same backend answers after a handful of
   Wikidata triples for the question's entity are placed in the context.

Each configuration's answers are written to ``results/rag_<backend>_<config>.json``
together with the gold answers, and a short summary table is printed.

Usage:
    python -m rag.run --backend extractive
    python -m rag.run --backend hf --model google/flan-t5-base
    python -m rag.run --backend openai --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .answer import answer_question
from .grade import is_correct, is_hallucination
from .retrieve import retrieve_triples

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "rag" / "data" / "questions.json"
RESULTS_DIR = ROOT / "results"


def evaluate(backend: str, model: str, with_retrieval: bool) -> dict:
    with open(QUESTIONS, "r", encoding="utf-8") as fh:
        questions = json.load(fh)["items"]

    records, correct, hallucinated, retrieval_misses = [], 0, 0, 0
    for q in questions:
        triples = None
        if with_retrieval:
            try:
                triples = retrieve_triples(
                    entity_qid=q["wikidata_entity"],
                    prefer_relation=q.get("relation"),
                )
            except Exception as exc:  # noqa: BLE001
                triples = []
                print(f"[warn] retrieval failed for {q['id']}: {exc}")
            if not triples:
                retrieval_misses += 1

        gen = answer_question(q["question"], triples, backend=backend, model_id=model)

        ok = is_correct(gen.answer, q["gold"])
        hl = is_hallucination(gen.answer, q["gold"])
        correct += int(ok)
        hallucinated += int(hl)

        records.append({
            "id": q["id"],
            "question": q["question"],
            "gold": q["gold"],
            "triples": [t.as_text() for t in (triples or [])],
            "answer": gen.answer,
            "correct": ok,
            "hallucination": hl,
        })

    n = len(questions)
    summary = {
        "backend": backend,
        "model": model,
        "with_retrieval": with_retrieval,
        "n_questions": n,
        "n_correct": correct,
        "n_hallucinated": hallucinated,
        "n_retrieval_misses": retrieval_misses,
        "accuracy": correct / n,
        "hallucination_rate": hallucinated / n,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "records": records,
    }
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["extractive", "hf", "openai"], default="extractive")
    p.add_argument("--model", default="google/flan-t5-base")
    args = p.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Backend = {args.backend}, model = {args.model}")
    print("-- Configuration 1: bare (no retrieval) --")
    bare = evaluate(args.backend, args.model, with_retrieval=False)
    print(f"  correct = {bare['n_correct']}/{bare['n_questions']}, "
          f"hallucinated = {bare['n_hallucinated']}")

    print("-- Configuration 2: KG-grounded (RAG) --")
    rag = evaluate(args.backend, args.model, with_retrieval=True)
    print(f"  correct = {rag['n_correct']}/{rag['n_questions']}, "
          f"hallucinated = {rag['n_hallucinated']}, "
          f"retrieval-misses = {rag['n_retrieval_misses']}")

    for tag, payload in (("bare", bare), ("rag", rag)):
        out = RESULTS_DIR / f"rag_{args.backend}_{tag}.json"
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
