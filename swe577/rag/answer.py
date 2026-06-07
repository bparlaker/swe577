"""Answer a factual question, with or without retrieved KG triples.

Three backends are supported, picked with the ``--backend`` flag:

* ``extractive``: deterministic, no LLM. With KG triples in context, looks
  for the gold-pattern object string within the triples. Without triples,
  always returns "I don't know" - the zero-knowledge baseline.
* ``hf``: a local Hugging Face seq2seq model (default: google/flan-t5-base).
  Runs on CPU; downloads weights on first use.
* ``openai``: any OpenAI-compatible chat-completion endpoint. Requires the
  ``OPENAI_API_KEY`` environment variable (and optionally ``OPENAI_BASE_URL``
  for local proxies like Ollama or vLLM).

This split lets the experiment be reproduced honestly: the ``extractive``
backend never hits an external LLM, so the comparison "bare model vs. KG
retrieval" stays clean and fully deterministic if so chosen.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .retrieve import Triple


PROMPT_WITH_CONTEXT = """Answer the question using only the facts in the
context below. If the facts do not contain the answer, reply exactly
"NOT FOUND". Keep the answer as short as possible.

Context:
{context}

Question: {question}
Answer:"""

PROMPT_BARE = """Answer the question as briefly and accurately as you can.
If you do not know the answer, reply exactly "I don't know".

Question: {question}
Answer:"""


@dataclass
class GenerationResult:
    answer: str
    used_triples: bool
    backend: str


# ----------------------------------------------------------------------
# extractive backend
# ----------------------------------------------------------------------
def _extractive_answer(question: str, triples: list[Triple] | None) -> str:
    """Very simple deterministic 'extractor': returns the object of the most
    relevant triple, judged by overlap of question tokens with the predicate
    label. With no triples available, returns a clear no-answer string."""
    if not triples:
        return "I don't know"

    q_tokens = set(re.findall(r"[a-z]+", question.lower()))
    best_score, best = -1, None
    for t in triples:
        p_tokens = set(re.findall(r"[a-z]+", t.predicate_label.lower()))
        overlap = len(q_tokens & p_tokens)
        if overlap > best_score:
            best_score, best = overlap, t

    return best.object_label if best else "NOT FOUND"


# ----------------------------------------------------------------------
# hugging face backend (lazy import)
# ----------------------------------------------------------------------
_HF_PIPE = None

def _hf_answer(prompt: str, model_id: str) -> str:
    global _HF_PIPE
    if _HF_PIPE is None:
        from transformers import pipeline as hf_pipeline  # noqa
        _HF_PIPE = hf_pipeline("text2text-generation", model=model_id)
    out = _HF_PIPE(prompt, max_new_tokens=40, do_sample=False)
    return out[0]["generated_text"].strip()


# ----------------------------------------------------------------------
# openai backend (lazy import)
# ----------------------------------------------------------------------
def _openai_answer(prompt: str, model_id: str) -> str:
    from openai import OpenAI  # noqa
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=40,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


# ----------------------------------------------------------------------
# public entry point
# ----------------------------------------------------------------------
def answer_question(
    question: str,
    triples: list[Triple] | None,
    backend: str = "extractive",
    model_id: str = "google/flan-t5-base",
) -> GenerationResult:
    if backend == "extractive":
        return GenerationResult(_extractive_answer(question, triples), bool(triples), backend)

    if triples:
        context = "\n".join(f"- {t.as_text()}" for t in triples)
        prompt = PROMPT_WITH_CONTEXT.format(context=context, question=question)
    else:
        prompt = PROMPT_BARE.format(question=question)

    if backend == "hf":
        text = _hf_answer(prompt, model_id)
    elif backend == "openai":
        text = _openai_answer(prompt, model_id)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return GenerationResult(text, bool(triples), backend)
