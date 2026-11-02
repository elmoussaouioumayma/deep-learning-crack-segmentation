"""Re-choose the decision threshold of a trained model on the validation images.

    python choose_threshold.py                     # updates weights/unet.pt in place

Use it when the threshold search range changed after training (the first CPU run only
searched up to 0.90, and 0.90 was always chosen). Only the 40 validation images are used,
never the test images.
"""

import argparse

import torch

from data import list_pairs, load_pair, split_train_val
from predict import load_model
from train import best_threshold, predict_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/DeepCrack")
    ap.add_argument("--weights", default="weights/unet.pt")
    a = ap.parse_args()
    model, old = load_model(a.weights)
    _, val_pairs = split_train_val(list_pairs(a.data, "train"))
    val = [load_pair(*p) for p in val_pairs]
    probs = predict_probs(model, [v[0] for v in val], "cpu")
    thr, f1 = best_threshold(probs, [v[1] for v in val])
    ck = torch.load(a.weights, map_location="cpu")
    ck.update(threshold=thr, val_f1=f1)
    torch.save(ck, a.weights)
    print(f"threshold {old:.2f} -> {thr:.2f}  (validation F1 {f1:.4f}, epoch {ck.get('epoch')})")


if __name__ == "__main__":
    main()
