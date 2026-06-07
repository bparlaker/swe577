# Experiment notes

This file documents the design decisions behind the two trials so that anyone
re-running the code (or the advisor reading the paper) can see exactly what
was done and why.

## Part 1: KGE reproduction (TransE vs. RotatE on FB15k-237 and WN18RR)

**Library.** The reproduction uses **pykeen 1.10+** rather than the original
RotatE author repository (`DeepGraphLearning/KnowledgeGraphEmbedding`). pykeen
provides the same TransE / RotatE models with the same loss functions
(MarginRankingLoss for TransE, Negative Sampling with Self-Adversarial / NSSA
loss for RotatE), and ships a unified rank-based evaluator with filtered
metrics. Using a maintained library keeps the code small and removes a class
of bugs that come from re-implementing training loops by hand.

**Datasets.** pykeen ships official loaders for FB15k-237 (Toutanova & Chen
2015) and WN18RR (Dettmers et al. 2018) and downloads them on first use into
`~/.data/pykeen`. The training, validation, and test splits are the standard
public ones; no resampling is done.

**Configurations.** Two configurations are provided per (model, dataset):

* `*_small.json` - reduced embedding dimension (200), reduced epoch count
  (50-80), and 32 negatives per positive. Designed to finish on CPU in
  20-45 minutes per run on a modern laptop. This is what is reported in
  Section 6 of the paper.
* `*_full.json` - the paper-faithful hyperparameters from Sun et al. 2019
  (embedding dim 1000 for FB15k-237 / 500 for WN18RR, 256-1024 negatives,
  1000 epochs). A GPU is needed for these to finish in any reasonable time.

**Evaluation.** pykeen's `RankBasedEvaluator(filtered=True)` is used, which
implements the standard "filtered" link-prediction protocol from Bordes et
al. 2013: for every test triple, all known true triples other than the test
triple are removed from the candidate ranking. Both head- and tail-side
predictions are averaged. Reported metrics are MRR and Hits@{1, 3, 10}.

**Expected behaviour.** Under either configuration, RotatE should outperform
TransE by a wider margin on WN18RR than on FB15k-237. The reason is that
WN18RR contains many symmetric and inverse relations (e.g. `_similar_to`,
`_member_of`), which TransE structurally cannot represent (since TransE
models a relation as a translation `h + r ≈ t`, a self-loop `h + r ≈ h`
would force `r ≈ 0`). RotatE models relations as rotations in complex space
and can express symmetry, inversion, and composition together in one model.
The numerical gap reproduces this expected qualitative behaviour.

## Part 2: KG-grounded RAG over Wikidata

**Question set.** Thirty hand-written factual questions about well-known
entities, each paired with a gold answer and the Wikidata Q-id of the
relevant entity. See `rag/data/questions.json`. The set is intentionally
biased towards questions whose answer is a single Wikidata property value,
so that the retrieval is straightforward and the comparison is meaningful.

**Retrieval.** `rag/retrieve.py` queries the public Wikidata SPARQL
endpoint (`https://query.wikidata.org/sparql`) for a small set of triples
about the entity. When the question item specifies a relation P-id, that
predicate is prioritised; otherwise a default set of high-utility predicates
(P36 capital, P37 official language, P50 author, ...) is used. The 0.3 s
delay between requests is the polite-use guideline recommended by Wikidata.

**Backends.** Three answering backends are provided:

* `extractive` - no LLM at all. With KG triples in context, picks the
  object of the triple whose predicate label has the highest token overlap
  with the question. Without triples, always replies "I don't know" (the
  zero-knowledge baseline). This backend exists so that the comparison can
  be reproduced bit-for-bit without any external dependency.
* `hf` - a local Hugging Face seq2seq model (default
  `google/flan-t5-base`). Free, runs on CPU.
* `openai` - any OpenAI-compatible chat-completion endpoint. Requires
  `OPENAI_API_KEY`.

**Grading.** `rag/grade.py` normalises both the gold and the model answer
(case folding, ASCII fold, whitespace) and marks the answer correct if the
gold string appears as a substring. A "hallucination" is a non-empty
answer that is wrong and is not an explicit abstention. The substring rule
is lenient on purpose; spot-check the per-question JSON output to confirm
the grading on borderline cases.

**Why this trial is small.** The 30-question set is not a benchmark; it is
a demonstration. Its purpose is to show, in a way that runs in minutes on
any laptop, the qualitative shift that KG grounding produces (fewer
hallucinations, errors move from invention to retrieval failure). Section 4
of the paper discusses the larger evaluation problem.

## Reproducibility

Every run is fully reproducible from a JSON config (for KGE) or a single
command (for RAG). All hyperparameters and the random seed are stored
alongside the metrics in `results/<run_name>.json`, so the exact run that
produced any number in the paper can be re-executed.
