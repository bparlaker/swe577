"""Retrieve triples from Wikidata for a given entity.

Two stages:

1. ``lookup_entity_qid`` takes an entity hint (a free-text name like
   "Turkey" or "Mona Lisa") and asks Wikidata's ``wbsearchentities`` API for
   the most likely Q-id. Doing the lookup at runtime, instead of relying on
   hard-coded identifiers, removes a whole class of silent failure: the
   question set does not need to be re-validated whenever Wikidata renames
   or re-numbers something.
2. ``retrieve_triples`` then pulls a small bundle of triples about that
   entity over SPARQL: the entity's label, plus a curated set of
   high-utility properties (capital, official language, author, director,
   creator, date of birth, etc.) that the question types in
   ``questions.json`` need answers to.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Optional

import requests

WBSEARCH_URL = "https://www.wikidata.org/w/api.php"
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = (
    "SWE577-Project/1.0 "
    "(https://github.com/<USERNAME>/swe577-kg-nlp-experiments; "
    "berker.parlaker@example.edu)"
)

# Properties pulled by default. Chosen to cover the question types in
# ``rag/data/questions.json`` (capital, language, currency, author,
# director, creator, founder, composer, chemical symbol/formula, date of
# birth, inventor, etc.).
DEFAULT_PROPERTIES = [
    "P31",   # instance of
    "P17",   # country
    "P36",   # capital
    "P37",   # official language
    "P38",   # currency
    "P50",   # author
    "P57",   # director
    "P61",   # discoverer or inventor
    "P86",   # composer
    "P98",   # editor
    "P112",  # founded by
    "P127",  # owned by
    "P131",  # located in administrative entity
    "P135",  # movement
    "P170",  # creator
    "P176",  # manufacturer
    "P246",  # element symbol
    "P274",  # chemical formula
    "P488",  # chairperson
    "P569",  # date of birth
    "P571",  # inception
    "P577",  # publication date
    "P800",  # notable work
    "P937",  # work location
]


@dataclass
class Triple:
    subject_qid: str
    subject_label: str
    predicate_pid: str
    predicate_label: str
    object_label: str

    def as_text(self) -> str:
        return f"({self.subject_label}, {self.predicate_label}, {self.object_label})"


def _http_get(url: str, params: dict, polite_delay: float = 0.3) -> dict:
    time.sleep(polite_delay)  # Wikidata etiquette
    headers = {
        "Accept": "application/sparql-results+json,application/json",
        "User-Agent": USER_AGENT,
    }
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def lookup_entity_qid(name: str, lang: str = "en") -> Optional[str]:
    """Return the top Wikidata Q-id for a free-text entity name, or None."""
    payload = _http_get(WBSEARCH_URL, {
        "action": "wbsearchentities",
        "search": name,
        "language": lang,
        "format": "json",
        "limit": 1,
        "type": "item",
    })
    hits = payload.get("search", [])
    return hits[0]["id"] if hits else None


def retrieve_triples(
    entity_qid: Optional[str] = None,
    entity_hint: Optional[str] = None,
    extra_relations: Iterable[str] = (),
    max_triples: int = 15,
) -> list[Triple]:
    """Return up to ``max_triples`` triples for an entity.

    Provide either ``entity_qid`` directly, or an ``entity_hint`` (an
    English name) that will be resolved to a Q-id via Wikidata's search
    API.
    """
    if entity_qid is None:
        if entity_hint is None:
            raise ValueError("Provide entity_qid or entity_hint")
        entity_qid = lookup_entity_qid(entity_hint)
        if entity_qid is None:
            return []  # no entity found in Wikidata

    pids: list[str] = []
    for pid in (*extra_relations, *DEFAULT_PROPERTIES):
        if pid not in pids:
            pids.append(pid)

    values = " ".join(f"wd:{p}" for p in pids)
    sparql = f"""
    SELECT ?p ?pLabel ?o ?oLabel ?sLabel WHERE {{
      VALUES ?p {{ {values} }}
      wd:{entity_qid} ?p ?o .
      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "en" .
        wd:{entity_qid} rdfs:label ?sLabel .
        ?p rdfs:label ?pLabel .
        ?o rdfs:label ?oLabel .
      }}
    }}
    LIMIT {max_triples}
    """

    payload = _http_get(SPARQL_ENDPOINT, {"query": sparql, "format": "json"})

    triples: list[Triple] = []
    for binding in payload.get("results", {}).get("bindings", []):
        pid = binding["p"]["value"].rsplit("/", 1)[-1]
        triples.append(Triple(
            subject_qid=entity_qid,
            subject_label=binding.get("sLabel", {}).get("value", entity_qid),
            predicate_pid=pid,
            predicate_label=binding.get("pLabel", {}).get("value", pid),
            object_label=binding.get("oLabel", {}).get("value", binding["o"]["value"]),
        ))
    return triples
