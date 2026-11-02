"""Compare the U-Net with the classical baseline on the 237 DeepCrack test images.

    python evaluate.py

Neither method has seen the test images: the U-Net was trained on the training images
(threshold chosen on the validation images) and the classical parameters were tuned on
training images. Writes results/test_metrics.json and the figures in images/.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from classical import ClassicalParams, segment
from data import download_deepcrack, list_pairs, load_pair
from metrics import counts, evaluate, prf
from predict import load_model, overlay, predict_probs


def summary(res_tol, res_strict):
    return {"precision": round(res_tol["precision"], 3), "recall": round(res_tol["recall"], 3),
            "f1_tol2": round(res_tol["f1"], 3), "f1_strict": round(res_strict["f1"], 3),
            "mean_iou": round(res_strict["mean_iou"], 3)}


def bootstrap_f1_diff(preds_a, preds_b, trues, tol=2, n=2000, seed=0):
    """Paired bootstrap over test images: 95 % interval of pooled F1(a) - F1(b)."""
    ca = np.array([counts(p, t, tol) for p, t in zip(preds_a, trues)])
    cb = np.array([counts(p, t, tol) for p, t in zip(preds_b, trues)])
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(trues), len(trues))
        diffs.append(prf(ca[idx].sum(0))[2] - prf(cb[idx].sum(0))[2])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return round(float(lo), 3), round(float(hi), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--weights", default="weights/unet.pt")
    ap.add_argument("--classical", default="results/classical_params.json")
    ap.add_argument("--figures", default="images")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = download_deepcrack(a.data)
    test = [load_pair(*p) for p in list_pairs(root, "test")]
    names = [p[0].stem for p in list_pairs(root, "test")]
    imgs, trues = [t[0] for t in test], [t[1] for t in test]

    model, thr = load_model(a.weights, device)
    probs = [predict_probs(model, im, device) for im in imgs]
    unet = [p > thr for p in probs]
    saved = json.loads(Path(a.classical).read_text())
    cp = ClassicalParams(**{k: saved[k] for k in ("kernel", "sigma", "k", "min_area", "min_elong")})
    classic = [segment(im, cp) for im in imgs]

    res = {}
    for name, preds in [("classical", classic), ("unet", unet)]:
        res[name] = (evaluate(preds, trues, tol=2), evaluate(preds, trues, tol=0))
    out = {k: summary(*v) for k, v in res.items()}
    f_u, f_c = np.array(res["unet"][0]["per_image_f1"]), np.array(res["classical"][0]["per_image_f1"])
    out["n_test_images"] = len(imgs)
    out["unet_better_on_images"] = int((f_u > f_c).sum())
    out["threshold"] = thr
    out["f1_tol2_diff_unet_minus_classical_95ci"] = bootstrap_f1_diff(unet, classic, trues)
    out["median_per_image_f1"] = {"classical": round(float(np.median(f_c)), 3), "unet": round(float(np.median(f_u)), 3)}
    Path("results").mkdir(exist_ok=True)
    Path("results/test_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    figs = Path(a.figures)
    figs.mkdir(exist_ok=True)

    # 1) metric comparison
    labels = ["Precision", "Recall", "F1 (2 px tol.)", "F1 (strict)", "mean IoU"]
    keys = ["precision", "recall", "f1_tol2", "f1_strict", "mean_iou"]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, (m, c, lab) in enumerate([("classical", "#9aa5b1", "Classical (black-hat + threshold)"),
                                     ("unet", "#d6452f", "U-Net (this repo)")]):
        v = [out[m][k] for k in keys]
        bars = ax.bar(x + (i - 0.5) * 0.38, v, 0.38, color=c, label=lab)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.18)
    ax.set_title(f"DeepCrack test set ({len(imgs)} images, unseen during training and tuning)")
    ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=9)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figs / "metrics.png", dpi=130)
    plt.close(fig)

    # 2) per-image F1
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.scatter(f_c, f_u, s=12, alpha=0.6, color="#d6452f")
    ax.plot([0, 1], [0, 1], color="#555", lw=1, ls="--")
    ax.set_xlabel("F1 classical")
    ax.set_ylabel("F1 U-Net")
    ax.set_title(f"Per test image: U-Net better on {out['unet_better_on_images']} / {len(imgs)}", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(figs / "per_image_f1.png", dpi=130)
    plt.close(fig)

    # 3) examples: typical cases (around the median gain) and the hardest cases for the U-Net
    gain = f_u - f_c
    order = np.argsort(gain)
    typical = [order[int(q * (len(order) - 1))] for q in (0.35, 0.6, 0.85)]
    worst = list(np.argsort(f_u)[:3])
    for fname, idx, title in [("examples.png", typical, "Test images"),
                              ("failure_cases.png", worst, "Hardest test images for the U-Net")]:
        fig, axes = plt.subplots(len(idx), 4, figsize=(13, 2.6 * len(idx)))
        for r, i in enumerate(idx):
            panels = [(imgs[i], "Image"), (trues[i], "Ground truth"),
                      (overlay(imgs[i], classic[i]), f"Classical  F1 {f_c[i]:.2f}"),
                      (overlay(imgs[i], unet[i]), f"U-Net  F1 {f_u[i]:.2f}")]
            for c, (im, t) in enumerate(panels):
                ax = axes[r, c]
                ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
                ax.set_title(t, fontsize=9)
                ax.axis("off")
        fig.suptitle(f"{title} (DeepCrack, Liu et al. 2019)", fontsize=11)
        fig.tight_layout()
        fig.savefig(figs / fname, dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
