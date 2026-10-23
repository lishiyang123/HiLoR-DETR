# Detector-dependent CCVR gains

CDW1630 Test: 237 images and 1,071 GT, detector seed 0 for DEIM and GCA.
Each detector uses its own frozen mapping and value-head fit.

| AP increment (pp) | DEIM | GCA |
|---|---:|---:|
| Competition mapping | 0.000000 | +0.176168 |
| Additional value update and competition recomputation | +0.028877 | +0.261200 |
| Complete CCVR | +0.028877 | +0.437368 |

The fitted kappa values are 0 and 0.5720062188069506. Full coefficients are in
`results/coefficients.csv`. Stage AP/AR values are fractions; increments are percentage points.

The diagnostic uses saved formal predictions, not a separate detector capture.
Before/after raw boxes, logits, decoder features and image sizes are identical
within each detector. DEIM mapping-only uses CPU FP32 replay through the shared
`competition_mapping.py`; the remaining stage values are from formal predictions.

| IoU | DEIM matched GT, before -> after | GCA matched GT, before -> after |
|---|---:|---:|
| 0.75 | 999 -> 999 | 1029 -> 1029 |
| 0.80 | 925 -> 926 | 983 -> 983 |
| 0.85 | 826 -> 827 | 893 -> 893 |
| 0.90 | 627 -> 628 | 702 -> 701 |
| 0.95 | 239 -> 240 | 312 -> 312 |

Matching is COCO one-to-one matching, faster-coco-eval 1.7.2, all object areas,
maxDets=1/10/100. The lost GCA match at IoU 0.90 is GT 585 in image 146:
its qualifying candidate remains in the global Top-300, but its within-image
category rank changes from 99 to 109, outside COCO's per-category top-100 limit.
The other four GCA matched-GT sets are unchanged while AP improves at all five thresholds.
This is a descriptive scoring/ranking diagnostic, not a causal decomposition of AP.

## CPU reproduction

```bash
python -B main_host/analysis/ccvr/run.py --release . --output /new/external/ccvr_seed0
```

Use the main-host CPU environment, code + results + the four shared seed-0 raw
Test captures in the optional features downloads. No detector weights, images,
detector forward, new fitting or GPU are needed. Inputs are referenced, not duplicated.

`results/matched_gt.csv` includes every GT, including unmatched ones.
Blank ranks mean unmatched. `rank` is dataset-wide within one category, after
COCO's per-image/per-category top-100 limit; it is not an image-level Top-300 rank.
Negative paired rank or preceding-FP changes indicate improvement. The full
summary and per-class threshold results are retained alongside the compact tables.
