# CDW1630 class-wise metrics

This analysis derives class-wise AP, AP75 and AR100 from the six frozen COCO
evaluation payloads already included in the package. It does not load detector
checkpoints and does not rerun training or inference.

From the package root, run:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python \
  analysis/classwise_metrics/extract_classwise_metrics.py
```

The Python environment must provide PyTorch, NumPy and the `faster_coco_eval`
package used to serialize `eval.pth`. The formal reproduction environment
already provides these dependencies.

Inputs:

- `datasets/CDW1630/annotations/instances_test2017.json`
- `results/CDW1630/B0/seed0-2/official_test/eval.pth`
- `results/CDW1630/Ours/seed0-2/final/eval.pth`

Outputs are written once to `results/CDW1630/classwise_metrics/`. Existing
outputs are never overwritten. All AP/AR columns are percentages; paired
deltas are percentage points. Three-seed dispersion uses the sample standard
deviation (`ddof=1`).
