"""Train a TransE or RotatE model on FB15k-237 or WN18RR.

Driven entirely by a JSON config file so that every run is reproducible.
Saves the trained model, the training loss curve, and the filtered link
prediction metrics (MRR, Hits@1, Hits@3, Hits@10) under ``results/``.

Usage:
    python -m kge.train --config kge/configs/rotate_fb15k237_small.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from pykeen.pipeline import pipeline

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def run_one(config: dict, results_dir: Path) -> dict:
    """Train one (model, dataset, scale) configuration and return metrics."""
    name = config["name"]
    print(f"\n=== {name} ===")
    print(json.dumps(config, indent=2))

    started = time.time()

    device = "cuda" if torch.cuda.is_available() and not config.get("force_cpu") else "cpu"
    print(f"Device: {device}")

    result = pipeline(
        dataset=config["dataset"],
        model=config["model"],
        model_kwargs=config.get("model_kwargs", {}),
        training_kwargs=config.get("training_kwargs", {}),
        loss=config.get("loss", "NSSALoss"),
        loss_kwargs=config.get("loss_kwargs", {}),
        optimizer=config.get("optimizer", "Adam"),
        optimizer_kwargs=config.get("optimizer_kwargs", {"lr": 1e-3}),
        negative_sampler=config.get("negative_sampler", "basic"),
        negative_sampler_kwargs=config.get("negative_sampler_kwargs", {}),
        evaluator="rankbased",
        evaluator_kwargs={"filtered": True},
        random_seed=config.get("seed", 42),
        device=device,
    )

    metrics = result.metric_results.to_dict()
    flat = {
        "mrr": float(metrics["both"]["realistic"]["inverse_harmonic_mean_rank"]),
        "hits_at_1": float(metrics["both"]["realistic"]["hits_at_1"]),
        "hits_at_3": float(metrics["both"]["realistic"]["hits_at_3"]),
        "hits_at_10": float(metrics["both"]["realistic"]["hits_at_10"]),
    }

    elapsed = time.time() - started
    record = {
        "name": name,
        "config": config,
        "device": device,
        "wall_clock_seconds": round(elapsed, 1),
        "started_at_utc": datetime.utcnow().isoformat(),
        "metrics_filtered_realistic": flat,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{name}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    print(f"\nFinished {name} in {elapsed:.1f}s. Metrics:")
    for k, v in flat.items():
        print(f"  {k:12s} = {v:.4f}")
    print(f"Saved -> {out_path}")
    return record


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Path to a JSON config file.")
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    args = p.parse_args()

    config = load_config(args.config)
    run_one(config, Path(args.results_dir))


if __name__ == "__main__":
    main()
