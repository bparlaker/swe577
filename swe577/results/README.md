# Results

This directory holds the JSON files written by `kge/train.py` and `rag/run.py`,
plus a small aggregator that produces the table used in Section 6 of the
paper.

Each KGE run writes a file named `<config_name>.json` (e.g.
`rotate_fb15k237_small.json`) with the full config, runtime, and the four
filtered link-prediction metrics (MRR, Hits@1, Hits@3, Hits@10).

Each RAG run writes two files, one per configuration:

* `rag_<backend>_bare.json` - bare model, no retrieval.
* `rag_<backend>_rag.json`  - KG-grounded model.

To collect everything into one table:

```bash
python -m results.aggregate
```

The `.json` files are gitignored, since they are reproduced by re-running
the experiments. The `summary.md` table is checked in so that anyone can
read the headline numbers without running the code.
