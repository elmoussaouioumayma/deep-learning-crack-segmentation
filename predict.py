"""Run the trained U-Net on any image.

    python predict.py photo.jpg                 # writes photo_crack_mask.png and photo_crack_overlay.png
    python predict.py photos/*.jpg --out masks

Images of any size work: they are padded to a multiple of 16 and large images are
processed in overlapping tiles, so memory use stays small.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from model import UNet

TILE, OVERLAP = 512, 64


def load_model(path="weights/unet.pt", device="cpu"):
    ck = torch.load(path, map_location=device)
    model = UNet(base=ck["base"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, float(ck.get("threshold", 0.5))


def _pad16(x):
    h, w = x.shape[:2]
    ph, pw = (-h) % 16, (-w) % 16
    return np.pad(x, ((0, ph), (0, pw), (0, 0)), mode="reflect"), h, w


@torch.no_grad()
def _run(model, rgb, device):
    x, h, w = _pad16(rgb)
    t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1), dtype=np.float32) / 255.0)[None].to(device)
    return torch.sigmoid(model(t))[0, 0].cpu().numpy()[:h, :w]


def predict_probs(model, rgb, device="cpu"):
    """Crack probability (0..1) for every pixel of an RGB uint8 image."""
    H, W = rgb.shape[:2]
    if max(H, W) <= TILE + OVERLAP:
        return _run(model, rgb, device)
    prob = np.zeros((H, W), np.float32)
    weight = np.zeros((H, W), np.float32)
    step = TILE - OVERLAP
    for y in list(range(0, max(H - TILE, 0), step)) + [max(H - TILE, 0)]:
        for x in list(range(0, max(W - TILE, 0), step)) + [max(W - TILE, 0)]:
            tile = rgb[y:y + TILE, x:x + TILE]
            prob[y:y + tile.shape[0], x:x + tile.shape[1]] += _run(model, tile, device)
            weight[y:y + tile.shape[0], x:x + tile.shape[1]] += 1
    return prob / weight


def overlay(rgb, mask, color=(255, 40, 40)):
    out = rgb.copy()
    out[mask.astype(bool)] = (0.35 * out[mask.astype(bool)] + 0.65 * np.array(color)).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--weights", default="weights/unet.pt")
    ap.add_argument("--threshold", type=float, default=None, help="default: the one chosen on validation data")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, thr = load_model(a.weights, device)
    thr = a.threshold if a.threshold is not None else thr
    for p in map(Path, a.images):
        rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        mask = predict_probs(model, rgb, device) > thr
        out = Path(a.out) if a.out else p.parent
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / f"{p.stem}_crack_mask.png"), mask.astype(np.uint8) * 255)
        cv2.imwrite(str(out / f"{p.stem}_crack_overlay.png"), cv2.cvtColor(overlay(rgb, mask), cv2.COLOR_RGB2BGR))
        print(f"{p.name}: {100 * mask.mean():.2f} % of the pixels are crack")


if __name__ == "__main__":
    main()
