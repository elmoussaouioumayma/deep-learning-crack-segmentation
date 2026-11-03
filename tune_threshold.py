"""Choose the decision threshold of a trained model on the validation images again.

    python tune_threshold.py --weights weights/unet.pt

Useful after changing the threshold search range: the model is not retrained, and the
threshold is chosen on the 40 validation images only (never on the test images).
"""

import argparse

import torch

from data import download_deepcrack, list_pairs, load_pair, split_train_val
from model import UNet
from train import best_threshold, predict_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/unet.pt")
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.weights, map_location=device)
    model = UNet(base=ck["base"]).to(device)
    model.load_state_dict(ck["state_dict"])
    _, val_pairs = split_train_val(list_pairs(download_deepcrack(a.data), "train"))
    val = [load_pair(*p) for p in val_pairs]
    probs = predict_probs(model, [v[0] for v in val], device)
    thr, f1 = best_threshold(probs, [v[1] for v in val])
    print(f"old threshold {ck['threshold']:.2f} (val F1 {ck['val_f1']:.3f}) -> new {thr:.2f} (val F1 {f1:.3f})")
    ck.update(threshold=thr, val_f1=f1)
    torch.save(ck, a.weights)


if __name__ == "__main__":
    main()
