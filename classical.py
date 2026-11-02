"""Classical crack segmentation baseline (no learning), tuned on the training images.

Pipeline: grey value -> local contrast (CLAHE) -> Gaussian smoothing -> black-hat filter
(highlights dark structures narrower than the kernel) -> threshold (Otsu, or mean + k
standard deviations of the black-hat response) -> removal of small and round blobs.
The parameters are chosen by grid search on the training images, with the same F1
metric as the U-Net, so the comparison is fair.
"""

import itertools
from concurrent.futures import ProcessPoolExecutor
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


def blackhat_response(rgb, kernel, sigma):
    """Black-hat response: bright where the surface has dark structures narrower than the kernel."""
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    grey = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grey)
    if sigma > 0:
        grey = cv2.GaussianBlur(grey, (0, 0), sigma)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    return cv2.morphologyEx(grey, cv2.MORPH_BLACKHAT, se)


def threshold_response(resp, k):
    if k is None:
        return cv2.threshold(resp, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return (resp > resp.mean() + k * resp.std()).astype(np.uint8)


def blob_stats(mask):
    """Connected components of the mask with their area and elongation (major / minor axis)."""
    lab = measure.label(mask)
    area = np.zeros(lab.max() + 1)
    elong = np.zeros(lab.max() + 1)
    for r in measure.regionprops(lab):
        area[r.label] = r.area
        elong[r.label] = r.axis_major_length / max(r.axis_minor_length, 1e-6)
    return lab, area, elong


def keep_blobs(stats, min_area, min_elong):
    """Keep blobs that are large enough and elongated (or very large, e.g. branched crack networks)."""
    lab, area, elong = stats
    keep = (area >= min_area) & ((elong >= min_elong) | (area > 20 * min_area))
    keep[0] = False
    return keep[lab].astype(np.uint8)


def segment(rgb, p: ClassicalParams):
    mask = threshold_response(blackhat_response(rgb, p.kernel, p.sigma), p.k)
    return keep_blobs(blob_stats(mask), p.min_area, p.min_elong)


# The first, smaller grid had its best values on the edges (kernel 31, sigma 2, min_area 30, min_elong 3),
# so the grid was extended in every direction around them.
GRID = dict(kernel=[15, 21, 31, 41, 51], sigma=[1.0, 2.0, 3.0], k=[None, 1.5, 2.0, 2.5, 3.0],
            min_area=[10, 20, 30, 50], min_elong=[2.0, 3.0, 4.0, 5.0])


def _search(args):
    """All thresholds and blob filters for one (kernel, sigma): returns [(F1, params), ...]."""
    images, masks, kernel, sigma, grid, tol = args
    resps = [blackhat_response(im, kernel, sigma) for im in images]
    out = []
    for k in grid["k"]:
        stats = [blob_stats(threshold_response(r, k)) for r in resps]
        for min_area, min_elong in itertools.product(grid["min_area"], grid["min_elong"]):
            tot = np.zeros(4)
            for st, t in zip(stats, masks):
                tot += counts(keep_blobs(st, min_area, min_elong), t, tol)
            out.append((prf(tot)[2], ClassicalParams(kernel, sigma, k, min_area, min_elong)))
    return out


def tune(images, masks, tol=2, grid=GRID, workers=None):
    """Grid search over the parameters (in parallel); returns the best ClassicalParams, its F1 and all results.

    The black-hat response is computed once per (kernel, sigma) and the blobs once per threshold,
    so the full grid stays cheap.
    """
    jobs = [(images, masks, kn, sg, grid, tol) for kn, sg in itertools.product(grid["kernel"], grid["sigma"])]
    with ProcessPoolExecutor(workers) as ex:
        results = [r for part in ex.map(_search, jobs) for r in part]
    f1, best = max(results, key=lambda r: r[0])
    return best, f1, results
