# Paired image Bootstrap

DEIM and HiLoR-DETR are compared separately for seeds 0, 1 and 2 on CDW1630
Test (237 images, 1,071 objects). Each pair uses 5,000 image resamples with
replacement, NumPy PCG64 seed 20260915, and a two-sided 95% percentile interval.
The same resampled images are used for both methods and across seeds. Per-image
matching is cached, but global COCO score ranking and accumulation are repeated
for every resample. AP is the primary metric; AP50, AP75 and AR100 are descriptive.

From the release root:

```bash
python -B bootstrap/run.py --output /new/external/bootstrap
```

Use the code and results downloads together: the entry reads the Test annotation
and six shared compressed predictions listed in `dataset_analysis/inputs.json`.
No model weights, images, feature captures, PyTorch or GPU are needed. The runtime
used was Python 3.11.9, NumPy 2.4.4 and faster-coco-eval 1.7.2. `--jobs 1` reduces
CPU concurrency; it does not change the sampling or results.

Results:

- `results/summary.csv`: original per-seed metrics and differences, confidence
  limits, bootstrap means/medians and descriptive sign-tail statistics.
- `results/seed_summary.csv`: the original three-seed means and sample SDs.
- `results/seed0/replicates.csv`, `seed1/replicates.csv`, `seed2/replicates.csv`:
  all 5,000 paired differences per seed.

Differences and interval limits are in percentage points, HiLoR minus DEIM.
The observed difference is not the bootstrap median. These intervals describe
image-sampling uncertainty conditional on each trained pair, not training-seed
uncertainty; ordinary image resampling does not correct for correlations among
related or near-duplicate Test images. Secondary metrics and sign-tail statistics
are descriptive, not an additional family of confirmatory tests.
