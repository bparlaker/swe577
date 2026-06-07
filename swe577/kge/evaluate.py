"""Re-evaluate a saved pykeen model on the test split.

Most users will just call ``kge.train``, which trains and evaluates in one
go. This helper exists for the case where a trained model is loaded from
disk and the metrics need to be recomputed (e.g. with a different
filtering strategy).

Usage:
    python -m kge.evaluate --model-path results/rotate_fb15k237_small/trained_model.pkl \\
                           --dataset FB15k237
"""

from __future__ import annotations

import argparse
import json

import torch
from pykeen.datasets import get_dataset
from pykeen.evaluation import RankBasedEvaluator


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--dataset", required=True)
    args = p.parse_args()

    model = torch.load(args.model_path, map_location="cpu")
    dataset = get_dataset(dataset=args.dataset)

    evaluator = RankBasedEvaluator(filtered=True)
    metrics = evaluator.evaluate(
        model=model,
        mapped_triples=dataset.testing.mapped_triples,
        additional_filter_triples=[
            dataset.training.mapped_triples,
            dataset.validation.mapped_triples,
        ],
    )
    print(json.dumps(metrics.to_dict(), indent=2))


if __name__ == "__main__":
    main()
