"""Retrieve triples from Wikidata for a given entity.

For every entity Q-id we pull the small bundle of triples that is most likely
to contain the answer: name + description + label + a handful of high-utility
property values (capital, head of state, official language, occupation,
country, etc.). When the question item names a specific relation P-id, we
prefer that relation; otherwise we fall back to a generic set.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import requests

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "SWE577-Project/1.0 (https://github.com/parlaker/swe577-kg-nlp; berker.parlaker@example.edu)"

# Properties pulled when no specific P-id is supplied for the question.
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
    "P112",  # founded by
    "P131",  # located in administrative entity
    "P170",  # creator
    "P246",  # element symbol
    "P488",  # chairperson
    "P569",  # date of birth
    "P571",  # inception
    "P176",  # manufacturer
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


def _query(sparql: str) -> dict:
    """Run a SPARQL query against Wikidata."""
    headers = {"Accept": "application/sparql-results+json", "User-Agent": USER_AGENT}
    response = requests.get(
        SPARQL_ENDPOINT,
        params={"query": sparql, "format": "json"},
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def retrieve_triples(
    entity_qid: str,
    prefer_relation: str | None = None,
    extra_relations: Iterable[str] = (),
    max_triples: int = 12,
    polite_delay: float = 0.3,
) -> list[Triple]:
    """Return up to ``max_triples`` triples for ``entity_qid`` from Wikidata."""
    pids = []
    if prefer_relation:
        pids.append(prefer_relation)
    for pid in (*extra_relations, *DEFAULT_PROPERTIES):
        if pid not in pids:
            pids.append(pid)

    values = " ".join(f"wd:{p}" for p in pids)
    query = f"""
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

    time.sleep(polite_delay)  # Wikidata's etiquette guideline
    payload = _query(query)

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
