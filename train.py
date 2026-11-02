"""Train the U-Net on DeepCrack.

    python train.py                         # downloads the data if needed, trains, saves the best model
    python train.py --epochs 80 --base 32   # bigger model / longer training (GPU, e.g. Google Colab)

The best model (highest F1 on the held-out validation images) is saved to
results/unet.pt together with the decision threshold chosen on the validation images.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data import augment, download_deepcrack, list_pairs, load_pair, split_train_val, to_tensor_arrays
from metrics import counts, prf
from model import UNet, count_parameters, dice_bce_loss


class CrackDataset(Dataset):
    def __init__(self, pairs, train=True, crop=256, repeats=1, seed=0):
        self.items = [load_pair(*p) for p in pairs]
        self.train, self.crop, self.repeats = train, crop, repeats
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items) * self.repeats

    def __getitem__(self, i):
        img, lab = self.items[i % len(self.items)]
        if self.train:
            img, lab = augment(img, lab, self.rng, self.crop)
        x, y = to_tensor_arrays(img, lab)
        return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def predict_probs(model, images, device, batch=4):
    """Crack probability map for full images (sizes must be multiples of 16)."""
    model.eval()
    out = []
    for i in range(0, len(images), batch):
        x = torch.from_numpy(np.stack([to_tensor_arrays(im)[0] for im in images[i:i + batch]])).to(device)
        out.extend(torch.sigmoid(model(x)).cpu().numpy()[:, 0])
    return out


def best_threshold(probs, masks, tol=2):
    """Threshold with the highest pooled F1 on the given images."""
    best = (0.5, -1.0)
    for t in np.arange(0.3, 0.981, 0.02):
        tot = np.zeros(4)
        for p, m in zip(probs, masks):
            tot += counts(p > t, m, tol)
        f = prf(tot)[2]
        if f > best[1]:
            best = (round(float(t), 2), f)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--base", type=int, default=16, help="channels of the first U-Net level")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--repeats", type=int, default=2, help="random crops per training image and epoch")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = download_deepcrack(a.data)
    train_pairs, val_pairs = split_train_val(list_pairs(root, "train"))
    train_ds = CrackDataset(train_pairs, train=True, repeats=a.repeats, seed=a.seed)
    val = [load_pair(*p) for p in val_pairs]
    loader = DataLoader(train_ds, batch_size=a.batch, shuffle=True, num_workers=2 if device == "cuda" else 0,
                        drop_last=True)

    model = UNet(base=a.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(loader))
    print(f"device {device} | U-Net base {a.base}: {count_parameters(model) / 1e6:.2f} M parameters | "
          f"{len(train_pairs)} train / {len(val_pairs)} validation images")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "training_log.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["epoch", "train_loss", "val_f1", "val_threshold", "seconds"])
    best_f1 = -1
    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = dice_bce_loss(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        probs = predict_probs(model, [v[0] for v in val], device)
        thr, f1 = best_threshold(probs, [v[1] for v in val])
        dt = time.time() - t0
        writer.writerow([epoch, round(float(np.mean(losses)), 4), round(f1, 4), thr, round(dt, 1)])
        log.flush()
        flag = ""
        if f1 > best_f1:
            best_f1 = f1
            torch.save({"state_dict": model.state_dict(), "base": a.base, "threshold": thr,
                        "val_f1": f1, "epoch": epoch}, out / "unet.pt")
            flag = "  <- best"
        print(f"epoch {epoch:3d}  loss {np.mean(losses):.3f}  val F1 {f1:.3f} (thr {thr:.2f})  {dt:.0f}s{flag}")
    log.close()
    (out / "training_summary.json").write_text(json.dumps({"best_val_f1": round(best_f1, 4), "epochs": a.epochs,
                                                           "base": a.base, "device": device}, indent=2))


if __name__ == "__main__":
    main()
