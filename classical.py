"""Classical crack segmentation baseline (no learning), tuned on the training images.

Pipeline: grey value -> local contrast (CLAHE) -> Gaussian smoothing -> black-hat filter
(highlights dark structures narrower than the kernel) -> threshold (Otsu, or mean + k
standard deviations of the black-hat response) -> removal of small and round blobs.
The parameters are chosen by grid search on the training images, with the same F1
metric as the U-Net, so the comparison is fair.
"""

import itertools
from dataclasses import dataclass

import cv2
import numpy as np
from skimage import measure

from metrics import counts, prf


@dataclass
class ClassicalParams:
    kernel: int = 15          # black-hat kernel (px): cracks narrower than this are kept
    sigma: float = 1.0        # Gaussian smoothing before the filter
    k: float = None           # None = Otsu threshold, else mean + k * std of the response
    min_area: int = 50        # remove blobs smaller than this (px)
    min_elong: float = 2.5    # remove blobs that are not elongated (major / minor axis)


def segment(rgb, p: ClassicalParams):
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    grey = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grey)
    if p.sigma > 0:
        grey = cv2.GaussianBlur(grey, (0, 0), p.sigma)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.kernel, p.kernel))
    resp = cv2.morphologyEx(grey, cv2.MORPH_BLACKHAT, se)
    if p.k is None:
        _, mask = cv2.threshold(resp, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        mask = (resp > resp.mean() + p.k * resp.std()).astype(np.uint8)
    lab = measure.label(mask)
    out = np.zeros_like(mask)
    for r in measure.regionprops(lab):
        if r.area < p.min_area:
            continue
        elong = r.axis_major_length / max(r.axis_minor_length, 1e-6)
        if elong >= p.min_elong or r.area > 20 * p.min_area:
            out[lab == r.label] = 1
    return out


COARSE_GRID = ([9, 15, 21, 31], [0.0, 1.0, 2.0], [None, 2.0, 3.0, 4.0], [30, 80, 150], [2.0, 3.0])


def tune(images, masks, tol=2, verbose=True, grid=COARSE_GRID):
    """Grid search over the parameters; returns the best ClassicalParams and its F1."""
    grid = itertools.product(*grid)
    best = (None, -1)
    for kernel, sigma, k, min_area, min_elong in grid:
        p = ClassicalParams(kernel, sigma, k, min_area, min_elong)
        tot = np.zeros(4)
        for img, m in zip(images, masks):
            tot += counts(segment(img, p), m, tol)
        f = prf(tot)[2]
        if f > best[1]:
            best = (p, f)
            if verbose:
                print(f"  F1 {f:.3f}  {p}")
    return best
