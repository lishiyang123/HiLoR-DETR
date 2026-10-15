# Full-query oracle geometry analysis

All values are GT-weighted; mean ± sample SD across detector seeds 0/1/2.

| Metric | DEIM | GCCS | Paired delta |
|---|---:|---:|---:|
| mean_best_iou | 0.905780 ± 0.001257 | 0.912200 ± 0.000441 | +0.006420 |
| median_best_iou | 0.922451 ± 0.001581 | 0.929682 ± 0.000800 | +0.007231 |
| oracle_coverage_iou_ge_075 | 97.665733 ± 0.280112 | 97.696857 ± 0.215630 | +0.031124 |

IoU values use the 0–1 scale. Coverage is shown as a percentage; its delta is in percentage points.

This is oracle geometric coverage, not actual detection recall: one query may independently cover multiple GTs.
No category, confidence, Top-K, NMS, or one-to-one matching is used. Results do not decompose AP gains.
CCVR retained the entire raw box pool bitwise unchanged for all 237 images of each GCCS seed.
All six checkpoint/data bindings and the detector source were verified unchanged after this analysis.
