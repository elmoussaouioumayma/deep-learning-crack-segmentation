"""Tune the classical baseline on training images and save its parameters.

    python tune_classical.py          # about 40 minutes on a laptop; writes results/classical_params.json

1. coarse grid search over all parameters
2. local refinement: while the best value of a parameter lies at the edge of what was
   tried, try its neighbours too, until the F1 stops improving. On DeepCrack the best
   black-hat kernel is much larger than in the coarse grid, because many cracks are wide.

Only training images are used (never the validation or test images), with the same
F1 metric as the U-Net, so the comparison is fair.
"""

import json
import random
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from classical import segment, tune
from data import list_pairs, load_pair, split_train_val
from metrics import counts, prf

STEPS = {"kernel": [-10, 10, 20], "sigma": [-1.0, 1.0], "k": [-0.5, 0.5], "min_area": [-15, 30], "min_elong": [-1.0, 1.0]}


def score(p, imgs, masks):
    tot = np.zeros(4)
    for img, m in zip(imgs, masks):
        tot += counts(segment(img, p), m, 2)
    return prf(tot)[2]


def refine(best, f1, imgs, masks, rounds=6):
    for _ in range(rounds):
        improved = False
        for name, steps in STEPS.items():
            cur = getattr(best, name)
            if cur is None:
                continue
            for s in steps:
                v = cur + s
                if v <= 0 or (name == "kernel" and v < 3):
                    continue
                cand = replace(best, **{name: int(v) if name in ("kernel", "min_area") else float(v)})
                f = score(cand, imgs, masks)
                if f > f1 + 1e-4:
                    best, f1, improved = cand, f, True
                    print(f"  F1 {f:.3f}  {best}")
        if not improved:
            break
    return best, f1


def main(root="data/DeepCrack", n_images=80, seed=0):
    train, _ = split_train_val(list_pairs(root, "train"))
    random.Random(seed).shuffle(train)
    imgs, masks = zip(*[load_pair(*p) for p in train[:n_images]])
    print("coarse grid")
    best, f1 = tune(imgs, masks)
    print("local refinement")
    best, f1 = refine(best, f1, imgs, masks)
    Path("results").mkdir(exist_ok=True)
    Path("results/classical_params.json").write_text(json.dumps({**asdict(best), "train_f1": round(f1, 4)}, indent=2))
    print("best:", best, "train F1", round(f1, 3))


if __name__ == "__main__":
    main()
