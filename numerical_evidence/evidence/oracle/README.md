# Oracle geometry

For every CDW1630 Test GT, take the maximum IoU over all 300 final-layer query
boxes, independently of class or score. No confidence filtering, Top-K, NMS,
clipping or one-to-one matching is used. DEIM and GSA use detector seeds 0/1/2;
each seed contains the same 237 images and 1,071 GT objects.

From the release root, recompute the results on CPU:

```bash
python -B numerical_evidence/evidence/oracle/run.py --output /new/external/oracle
```

Requires NumPy, the code and results downloads, and the supplied Test annotation.
No images, model weights, feature captures, PyTorch, COCO evaluator or GPU are
needed. The entry locates the release and its six compact box files automatically.
Use `--release` or `--boxes` to override those locations. Output must be a new
directory outside the release.

## Inputs and results

- `results/boxes/{deim,gsa}/seedN.npz`: full FP32 normalized cxcywh query boxes,
  image IDs and original sizes in height/width order. Class scores are not used.
- `results/per_gt.csv`: 3,213 paired GT rows, best IoU and zero-based argmax query
  IDs for both detectors.
- `results/per_seed.csv`: six model/seed rows, GT counts, mean/median best IoU,
  coverage at IoU >= 0.75 and nonpositive-area prediction counts.
- `results/summary.csv`: three-seed means, sample SDs and paired GSA-minus-DEIM
  changes for the three metrics.

Within each seed, every GT has equal weight; per-image means are not averaged.
The summary's median row averages the three per-seed medians. IoU and coverage
use the 0-1 scale in all CSVs. Multiply coverage and its change by 100 for percent
and percentage-point reporting. Coordinates and IoU use float64 arithmetic on
the saved FP32 detector outputs, without clipping. Nonpositive-area predictions
remain in the pool with IoU zero; nonfinite values stop computation.

To extract compact boxes from trusted full project captures instead, add
`--captures /path/to/work` (PyTorch required). The work directory contains
`native_seedN/evaluate_test/` and `gsa_seedN/evaluate_test/`, each with
`test_capture.pt` and `RESULT.json`. This mode reads CPU caches, not a detector.

Oracle geometric coverage is not detection recall or a TP count: one query can
be best for multiple GTs. It does not decompose AP gains. CCVR changes scores
and output selection rather than query-box coordinates, so no separate CCVR
geometric pool is needed. This entry does not execute CCVR or perform a runtime
coordinate-invariance test.
