# AP-IoU curves

DEIM and complete HiLoR-DETR on CDW1630 Test: seeds 0/1/2, 237 images and
1,071 GT objects. The ten IoU thresholds are 0.50 to 0.95 in steps of 0.05,
with standard COCO area=all and maxDets=100.

From the release root, recompute both panels on CPU:

```bash
python -B numerical_evidence/evidence/ap_iou_curve/run.py --output /new/external/ap_iou
```

Requires NumPy and faster-coco-eval 1.7.2, plus the code and results downloads.
The six shared formal predictions are listed in `dataset_analysis/inputs.json`.
No model weights, images, feature captures, PyTorch or GPU are needed. Output
must be a new directory outside the release. Use `--release` to override the
automatically located release root.

`results/curve.csv` contains all six per-seed curves, method means and sample
SDs, and within-seed HiLoR-minus-DEIM differences with their mean and sample SD.
AP columns use the 0-1 scale; multiply by 100 for a percent ordinate. Delta
columns already use percentage points. The right-panel band is the paired
mean +/- one sample SD across three seeds, not a confidence interval or a
Bootstrap result. Means and sample SDs use `statistics.fmean` and
`statistics.stdev`.

To read trusted saved evaluation arrays instead, add `--work /path/to/work`.
This optional mode requires PyTorch and `native_seedN/evaluate_test/` and
`full_seedN/evaluate_test/`, each containing `eval.pth` and `RESULT.json`.

The command does not train, fit, forward a detector or draw graphics. Strict-IoU
AP also depends on classes and ranking; it does not isolate geometric box
quality or decompose AP gains. The separate oracle analysis is in
`../oracle/`.
