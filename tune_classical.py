"""Tune the classical baseline on training images and save its parameters.

    python tune_classical.py          # writes results/classical_params.json
"""

import json
import random
from dataclasses import asdict
from pathlib import Path

from classical import GRID, tune
from data import list_pairs, load_pair, split_train_val


def main(root="data/DeepCrack", n_images=80, seed=0):
    train, _ = split_train_val(list_pairs(root, "train"))
    random.Random(seed).shuffle(train)
    imgs, masks = zip(*[load_pair(*p) for p in train[:n_images]])
    best, f1, results = tune(imgs, masks)
    Path("results").mkdir(exist_ok=True)
    Path("results/classical_params.json").write_text(json.dumps(
        {**asdict(best), "train_f1": round(f1, 4), "n_tuning_images": len(imgs), "grid": GRID,
         "n_combinations": len(results)}, indent=2))
    print("top 5:")
    for f, p in sorted(results, key=lambda r: -r[0])[:5]:
        print(f"  F1 {f:.4f}  {p}")
    print("best:", best, "train F1", round(f1, 3))


if __name__ == "__main__":
    main()
