"""Grade a generated answer against the gold answer.

The judgement is intentionally lenient and substring-based, since LLMs
phrase the right answer in many different ways. A small set of explicit
synonyms (e.g. "USA" / "United States") would normally live here too;
keep this trivial for the academic write-up so that the grading is
transparent and reproducible.
"""

from __future__ import annotations

import re
import unicodedata


def normalise(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_correct(generated: str, gold: str) -> bool:
    gen = normalise(generated)
    gold_n = normalise(gold)
    if not gold_n:
        return False
    return gold_n in gen


def is_hallucination(generated: str, gold: str) -> bool:
    """A confident-but-wrong answer: a non-empty answer that does not
    contain the gold substring and is not an explicit "don't know"."""
    if is_correct(generated, gold):
        return False
    n = normalise(generated)
    if not n:
        return False
    abstentions = {"i don t know", "not found", "unknown", "i dont know"}
    return n not in abstentions and not any(a in n for a in abstentions)
