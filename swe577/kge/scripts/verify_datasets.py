"""Force pykeen to download (and cache) FB15k-237 and WN18RR.

Useful as a one-shot step on a new machine so that the subsequent
training runs do not have to wait on network during their first epoch.
"""

from pykeen.datasets import get_dataset

for name in ("FB15k237", "WN18RR"):
    ds = get_dataset(dataset=name)
    print(f"{name}: train={ds.training.num_triples}, "
          f"valid={ds.validation.num_triples}, test={ds.testing.num_triples}, "
          f"entities={ds.num_entities}, relations={ds.num_relations}")
