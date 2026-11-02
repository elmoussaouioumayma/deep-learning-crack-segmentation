"""DeepCrack dataset: download, loading and augmentation.

DeepCrack (Liu et al., Neurocomputing 2019) has 537 RGB photos of cracks in concrete and
asphalt (544 x 384 px) with hand-drawn binary masks: 300 for training, 237 for testing.
The images are NOT stored in this repository; they are downloaded from the authors'
GitHub repository. Their use is restricted to non-commercial research and education.
"""

import random
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

URL = "https://github.com/yhlleo/DeepCrack/raw/master/dataset/DeepCrack.zip"


def download_deepcrack(root="data"):
    """Download and unpack DeepCrack into root/DeepCrack (train_img, train_lab, test_img, test_lab)."""
    root = Path(root)
    target = root / "DeepCrack"
    if (target / "test_lab").exists():
        return target
    root.mkdir(parents=True, exist_ok=True)
    zpath = root / "DeepCrack.zip"
    if not zpath.exists():
        try:
            urllib.request.urlretrieve(URL, zpath)
        except Exception:                                  # fall back to git if direct download is blocked
            tmp = root / "_DeepCrack_repo"
            subprocess.run(["git", "clone", "--depth", "1", "https://github.com/yhlleo/DeepCrack.git", str(tmp)],
                           check=True)
            shutil.copy(tmp / "dataset" / "DeepCrack.zip", zpath)
            shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(target)
    return target


def list_pairs(root, split):
    """(image_path, mask_path) pairs for split = 'train' or 'test'."""
    root = Path(root)
    pairs = []
    for img in sorted((root / f"{split}_img").glob("*.jpg")):
        lab = root / f"{split}_lab" / (img.stem + ".png")
        if lab.exists():
            pairs.append((img, lab))
    return pairs


def split_train_val(pairs, n_val=40, seed=0):
    """Hold out part of the official training set for model selection and threshold tuning."""
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    return sorted(pairs[n_val:]), sorted(pairs[:n_val])


def load_pair(img_path, lab_path):
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    lab = (cv2.imread(str(lab_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    return img, lab


def augment(img, lab, rng, crop=256):
    """Random crop, flips, 90-degree rotations, brightness/contrast/colour jitter, blur and noise."""
    H, W = lab.shape
    if crop and H >= crop and W >= crop:
        # prefer crops that contain cracks (they cover only ~3 % of the pixels)
        for _ in range(10):
            y, x = rng.integers(0, H - crop + 1), rng.integers(0, W - crop + 1)
            if lab[y:y + crop, x:x + crop].mean() > 0.005 or rng.random() < 0.2:
                break
        img, lab = img[y:y + crop, x:x + crop], lab[y:y + crop, x:x + crop]
    if rng.random() < 0.5:
        img, lab = img[:, ::-1], lab[:, ::-1]
    if rng.random() < 0.5:
        img, lab = img[::-1], lab[::-1]
    k = int(rng.integers(0, 4))
    img, lab = np.rot90(img, k), np.rot90(lab, k)
    img = img.astype(np.float32)
    img = img * rng.uniform(0.7, 1.3) + rng.uniform(-30, 30)                  # contrast, brightness
    img = img * rng.uniform(0.9, 1.1, size=(1, 1, 3))                        # colour cast
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.5))
    if rng.random() < 0.3:
        img = img + rng.normal(0, rng.uniform(2, 8), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), np.ascontiguousarray(lab)


def to_tensor_arrays(img, lab=None):
    """HWC uint8 -> CHW float32 in [0, 1] (ImageNet-style normalisation is not needed here)."""
    x = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return (x, None if lab is None else lab[None].astype(np.float32))
