"""Segmentation metrics for thin structures.

Crack masks are only a few pixels wide and hand annotation is not pixel-exact, so the
usual crack-detection metric allows a small tolerance: a predicted crack pixel is
correct if a true crack pixel lies within `tol` pixels, and vice versa for recall.
With tol = 0 the metric is the strict pixel-wise F1.
"""

import cv2
import numpy as np


def _dilate(mask, tol):
    if tol <= 0:
        return mask.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def counts(pred, true, tol=2):
    """(true positives for precision, predicted pixels, true positives for recall, true pixels)."""
    pred, true = pred.astype(bool), true.astype(bool)
    tp_p = (pred & _dilate(true, tol)).sum()
    tp_r = (true & _dilate(pred, tol)).sum()
    return int(tp_p), int(pred.sum()), int(tp_r), int(true.sum())


def prf(c):
    tp_p, n_pred, tp_r, n_true = c
    p = tp_p / n_pred if n_pred else (1.0 if n_true == 0 else 0.0)
    r = tp_r / n_true if n_true else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def iou(pred, true):
    pred, true = pred.astype(bool), true.astype(bool)
    union = (pred | true).sum()
    return (pred & true).sum() / union if union else 1.0


def evaluate(preds, trues, tol=2):
    """Dataset-level precision / recall / F1 (pixels pooled over all images) and mean IoU."""
    tot = np.zeros(4)
    ious, f1s = [], []
    for p, t in zip(preds, trues):
        c = counts(p, t, tol)
        tot += c
        f1s.append(prf(c)[2])
        ious.append(iou(p, t))
    P, R, F = prf(tot)
    return {"precision": P, "recall": R, "f1": F, "mean_iou": float(np.mean(ious)),
            "per_image_f1": f1s}
