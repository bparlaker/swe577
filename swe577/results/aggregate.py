"""Read every results/*.json file and produce a clean summary table."""

from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
KGE_PREFIXES = ("transe_", "rotate_")
RAG_PREFIX = "rag_"


def collect_kge() -> list[dict]:
    rows = []
    for f in sorted(HERE.glob("*.json")):
        if not f.name.startswith(KGE_PREFIXES):
            continue
        with open(f, "r", encoding="utf-8") as fh:
            r = json.load(fh)
        rows.append({
            "name": r["name"],
            "dataset": r["config"]["dataset"],
            "model": r["config"]["model"],
            "mrr": r["metrics_filtered_realistic"]["mrr"],
            "hits_at_1": r["metrics_filtered_realistic"]["hits_at_1"],
            "hits_at_3": r["metrics_filtered_realistic"]["hits_at_3"],
            "hits_at_10": r["metrics_filtered_realistic"]["hits_at_10"],
            "wall_s": r["wall_clock_seconds"],
        })
    return rows


def collect_rag() -> list[dict]:
    rows = []
    for f in sorted(HERE.glob(f"{RAG_PREFIX}*.json")):
        with open(f, "r", encoding="utf-8") as fh:
            r = json.load(fh)
        rows.append({
            "backend": r["backend"],
            "model": r["model"],
            "configuration": "rag" if r["with_retrieval"] else "bare",
            "n_questions": r["n_questions"],
            "n_correct": r["n_correct"],
            "n_hallucinated": r["n_hallucinated"],
            "n_retrieval_misses": r["n_retrieval_misses"],
            "accuracy": r["accuracy"],
            "hallucination_rate": r["hallucination_rate"],
        })
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_md(kge_rows: list[dict], rag_rows: list[dict], path: Path) -> None:
    lines = ["# Experiment summary\n\n## Knowledge-graph embedding\n"]
    if kge_rows:
        lines.append("| Dataset | Model | MRR | Hits@1 | Hits@3 | Hits@10 | Wall (s) |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for r in kge_rows:
            lines.append(
                f"| {r['dataset']} | {r['model']} | {r['mrr']:.3f} | "
                f"{r['hits_at_1']:.3f} | {r['hits_at_3']:.3f} | "
                f"{r['hits_at_10']:.3f} | {r['wall_s']:.0f} |"
            )
    else:
        lines.append("_(no KGE runs found)_")

    lines.append("\n## KG-grounded RAG\n")
    if rag_rows:
        lines.append("| Backend | Configuration | Correct/30 | Hallucinated | Retrieval misses |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for r in rag_rows:
            lines.append(
                f"| {r['backend']} | {r['configuration']} | "
                f"{r['n_correct']}/{r['n_questions']} | "
                f"{r['n_hallucinated']} | {r['n_retrieval_misses']} |"
            )
    else:
        lines.append("_(no RAG runs found)_")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    kge_rows = collect_kge()
    rag_rows = collect_rag()
    write_csv(kge_rows, HERE / "summary_kge.csv")
    write_csv(rag_rows, HERE / "summary_rag.csv")
    write_md(kge_rows, rag_rows, HERE / "summary.md")
    print(f"KGE rows: {len(kge_rows)}, RAG rows: {len(rag_rows)}")
    print(f"Wrote summary.md, summary_kge.csv, summary_rag.csv")


if __name__ == "__main__":
    main()
