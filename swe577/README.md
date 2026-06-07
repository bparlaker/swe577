# Hands-on experiments for: *The Use of Knowledge Graphs in Natural Language Processing*

Companion code and data for the SWE 577 Spring 2026 term paper by Berker Parlaker
(advisor: Suzan Üsküdarlı, Boğaziçi University). The paper surveys how knowledge
graphs (KGs) are combined with NLP, and reports two small hands-on trials. This
repository contains the code, configurations, and evaluation data for those
trials so that the results can be reproduced.

The repository has two self-contained parts:

1. **`kge/`** — a knowledge-graph embedding reproduction. Trains TransE and
   RotatE on the two standard link-prediction benchmarks **FB15k-237** and
   **WN18RR**, and reports filtered Mean Reciprocal Rank (MRR) and Hits@10.
   Built on the `pykeen` library, which provides the model implementations, the
   datasets (auto-downloaded), and the standard evaluation protocol.

2. **`rag/`** — a minimal KG-grounded Retrieval-Augmented Generation pipeline.
   Answers a set of factual questions about well-known entities, first from a
   bare language model and then with a few triples retrieved from **Wikidata**
   added to the context. Evaluation is by hand-graded correctness against a
   small held-out question set.

## Datasets

| Name | Source | Used in |
| --- | --- | --- |
| FB15k-237 | Toutanova & Chen, 2015 — A subset of Freebase with inverse relations removed. Auto-downloaded by pykeen from the original release. | `kge/` |
| WN18RR | Dettmers et al., 2018 — A subset of WordNet with leakage from inverse relations removed. Auto-downloaded by pykeen from the original release. | `kge/` |
| Wikidata (live) | https://query.wikidata.org/sparql — public SPARQL endpoint. Queried at runtime; nothing is downloaded in bulk. | `rag/` |
| `rag/data/questions.json` | Hand-curated 30-question factual QA set assembled for this project (see file header for sources and provenance). | `rag/` |

## Quick reproduction

```bash
# 1. Create a virtual env and install dependencies
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. KG-embedding reproduction (TransE + RotatE on FB15k-237 and WN18RR)
#    Reduced-scale configs that finish in ~30 min on a modern CPU per run.
python -m kge.train --config kge/configs/transe_fb15k237_small.json
python -m kge.train --config kge/configs/rotate_fb15k237_small.json
python -m kge.train --config kge/configs/transe_wn18rr_small.json
python -m kge.train --config kge/configs/rotate_wn18rr_small.json
# Full-scale configs (paper-faithful, GPU recommended) are also provided:
#   kge/configs/{transe,rotate}_{fb15k237,wn18rr}_full.json

# 3. RAG pipeline (requires network for Wikidata; no GPU needed)
python -m rag.run --backend extractive                # zero-LLM baseline
python -m rag.run --backend hf --model google/flan-t5-base   # local LLM
# (optional) python -m rag.run --backend openai --model gpt-4o-mini

# 4. Aggregate results into a single table
python -m results.aggregate
```

Each run writes a JSON file under `results/` with the configuration, the per-step
training loss (for `kge/`), and the final evaluation metrics. The aggregator
collects these into `results/summary.csv` and a Markdown table that is included
verbatim in the paper.

## Repository layout

```
.
├── README.md                  # this file
├── requirements.txt
├── LICENSE                    # MIT
├── .gitignore
├── docs/
│   └── experiment_notes.md    # design decisions, deviations from paper config
├── kge/
│   ├── __init__.py
│   ├── train.py               # main training entry point
│   ├── evaluate.py            # filtered MRR / Hits@K
│   ├── configs/               # JSON configs, one per (model, dataset, scale)
│   └── scripts/
│       └── verify_datasets.py # confirms pykeen has cached the datasets
├── rag/
│   ├── __init__.py
│   ├── run.py                 # main entry point
│   ├── retrieve.py            # Wikidata SPARQL retrieval
│   ├── answer.py              # prompt assembly + LLM call / extractive baseline
│   ├── grade.py               # exact-match + manual-review grading
│   └── data/
│       └── questions.json     # 30 factual questions with gold answers
└── results/
    ├── README.md
    └── aggregate.py
```

## Reproducing the numbers in the paper

The numbers in Section 6 of the term paper come from the **reduced-scale**
configurations (`*_small.json`) so that the whole pipeline runs in a few hours
on a single CPU. The trade-off is that the absolute scores sit a little below
the values in the original RotatE paper (Sun et al., 2019), which were obtained
with the full configurations (`*_full.json`) on GPU. The *relative ordering* of
RotatE vs. TransE is stable across the two scales, which is the point the
demonstration is designed to make.

If you want to reproduce the published RotatE numbers exactly, run the `*_full`
configurations on a GPU; the configurations match the hyperparameters reported
in Sun et al., 2019.

## Citations

Datasets and methods are cited in the paper. Key references:

- Sun, Z., Z.-H. Deng, J.-Y. Nie and J. Tang, "RotatE: Knowledge Graph
  Embedding by Relational Rotation in Complex Space," ICLR, 2019.
- Wang, Q., Z. Mao, B. Wang and L. Guo, "Knowledge Graph Embedding: A Survey
  of Approaches and Applications," IEEE TKDE, 29(12):2724–2743, 2017.
- Pan, S., et al., "Unifying Large Language Models and Knowledge Graphs: A
  Roadmap," IEEE TKDE, 36(7):3580–3599, 2024.
- Toutanova, K. and Chen, D., "Observed versus latent features for knowledge
  base and text inference," 3rd Workshop on Continuous Vector Space Models and
  their Compositionality, 2015. (FB15k-237)
- Dettmers, T., Minervini, P., Stenetorp, P. and Riedel, S., "Convolutional 2D
  Knowledge Graph Embeddings," AAAI, 2018. (WN18RR)
- Ali, M., et al., "PyKEEN 1.0: A Python Library for Training and Evaluating
  Knowledge Graph Embeddings," JMLR 22(82):1–6, 2021. (the library used here)

## License

MIT. See `LICENSE`.
