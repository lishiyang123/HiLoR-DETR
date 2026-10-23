#!/usr/bin/env python
"""Shared COCO-eval exact-metric extraction helpers (read-only).

Reads the raw COCOeval dict (as saved by capture.py as eval.pt) and extracts
high-precision AP/AR: standard 12, per-IoU, per-class, per-size.
"""
import numpy as np


def mean_valid(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[x > -1]
    if x.size == 0:
        return float("nan")
    return float(x.mean())


def extract_exact_metrics(e):
    precision = np.asarray(e["precision"], dtype=np.float64)
    recall = np.asarray(e["recall"], dtype=np.float64)
    p = e["params"]
    ious = np.asarray(p.iouThrs)
    areas = list(p.areaRngLbl)
    maxdets = list(p.maxDets)
    cat_ids = [int(c) for c in list(p.catIds)]
    # The CDW1630 official dataset has fixed 5 categories (0-based IDs).
    KNOWN_NAMES = ["Brick", "Concrete", "Plastic Bottle", "Reinforcement", "timber"]
    if getattr(p, "catNames", None):
        cat_names = list(p.catNames)
    elif len(cat_ids) == 5:
        cat_names = KNOWN_NAMES
    else:
        cat_names = [str(c) for c in cat_ids]

    def tiou(v):
        idx = np.where(np.isclose(ious, v))[0]
        if len(idx) != 1:
            return None
        return int(idx[0])

    def ai(name):
        return areas.index(name)

    def mi(n):
        return maxdets.index(n)

    def AP(iou=None, area="all", maxdet=100, cat=None):
        a = ai(area)
        m = mi(maxdet)
        if iou is None:
            if cat is None:
                x = precision[:, :, :, a, m]
            else:
                k = cat_ids.index(cat)
                x = precision[:, :, k, a, m]
        else:
            t = tiou(iou)
            if t is None:
                return float("nan")
            if cat is None:
                x = precision[t, :, :, a, m]
            else:
                k = cat_ids.index(cat)
                x = precision[t, :, k, a, m]
        return mean_valid(x)

    def AR(area="all", maxdet=100, cat=None):
        a = ai(area)
        m = mi(maxdet)
        if cat is None:
            x = recall[:, :, a, m]
        else:
            k = cat_ids.index(cat)
            x = recall[:, k, a, m]
        return mean_valid(x)

    overall = {
        "AP": AP(), "AP50": AP(0.50), "AP75": AP(0.75),
        "APs": AP(area="small"), "APm": AP(area="medium"), "APl": AP(area="large"),
        "AR1": AR(maxdet=1), "AR10": AR(maxdet=10), "AR100": AR(maxdet=100),
        "ARs": AR(area="small"), "ARm": AR(area="medium"), "ARl": AR(area="large"),
    }

    # extra per-IoU AP anchors used by the audit
    for v in [0.80, 0.85, 0.90, 0.95]:
        overall[f"AP{int(v*100)}"] = AP(v)

    per_iou_ap = {}
    per_iou_ar = {}
    for v in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        per_iou_ap[f"AP{v:.2f}"] = AP(v)
    for t, v in enumerate(ious):
        per_iou_ar[f"AR{v:.2f}"] = mean_valid(recall[t, :, ai("all"), mi(100)])

    per_class = {}
    for cid, cname in zip(cat_ids, cat_names):
        per_class[cname] = {
            "AP": AP(cat=cid),
            "AP50": AP(0.50, cat=cid),
            "AP75": AP(0.75, cat=cid),
            "AP80": AP(0.80, cat=cid),
            "AP85": AP(0.85, cat=cid),
            "AP90": AP(0.90, cat=cid),
            "AP95": AP(0.95, cat=cid),
            "AR100": AR(maxdet=100, cat=cid),
            "APs": AP(area="small", cat=cid),
            "APm": AP(area="medium", cat=cid),
            "APl": AP(area="large", cat=cid),
        }

    per_size = {}
    for s in ["small", "medium", "large"]:
        per_size[s] = {
            "AP": AP(area=s),
            "AR100": AR(area=s, maxdet=100),
        }

    return {
        "overall": overall,
        "per_iou_ap": per_iou_ap,
        "per_iou_ar": per_iou_ar,
        "per_class": per_class,
        "per_size": per_size,
        "cat_names": cat_names,
        "cat_ids": cat_ids,
    }
