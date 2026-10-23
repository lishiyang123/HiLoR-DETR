# Dataset statistics and Test analyses

All commands run on CPU and write to a new directory outside the release.

## Annotation statistics

```bash
python -B dataset_analysis/source/analyze_dataset.py --project . --output /new/external/data_stats
```

Uses COCO annotations and image paths under `main_host/datasets/CDW1630/`;
only Python's standard library is required. It does not read image pixels.
`tables/` contains split, category, bbox-scale, object-density-bin and per-image
counts. `results/` contains numerical and paper-facing annotation summaries.
Small boxes have area <32² pixels; medium boxes have area ≥32² and <96²;
large boxes have area ≥96². Density bins are 0, 1, 2, 3–5, 6–10 and >10.

## Condition groups and per-class metrics

```bash
python -B dataset_analysis/run.py --stage all --output /new/external/test_analysis
```

Use NumPy and faster-coco-eval 1.7.2 in the main-host environment. `--stage
groups` or `--stage classes` runs only that analysis. No GPU, detector weights,
image archive, feature captures, training or fitting is needed. One CPU thread
is used. The six formal prediction files for DEIM and HiLoR-DETR, seeds 0/1/2,
are shared under `predictions/` and losslessly compressed as JSON.gz.
`inputs.json` gives release-relative paths and full-Test reference metrics.

Density is the number of GT objects in an image, with a **fixed cutoff of 5**:
low density N_i ≤5, high density N_i >5. For overlap, each GT object's score
is its largest intersection-over-own-area with another GT box of any class;
the image score is the mean over objects, or zero for an empty/single-object
image. The overlap cutoff is the Train image-score median when positive,
otherwise its inclusive 75th percentile when positive, otherwise zero.
Low overlap is ≤ the cutoff, high overlap is > it. Each group contains whole
Test images and retains every GT object. `grouping/thresholds.json` gives the
rules and values; `grouping/test_image_groups.json` gives the image IDs.

`conditional_performance/` contains AP/AP50/AP75/AR100 for four groups, all
24 method/seed/group combinations, 12 paired seed differences, and four
three-seed summaries. Per-class AP/AP75/AR100 and GT counts are stored under
`../main_host/results/CDW1630/classwise_metrics/`. The reproduction command
regenerates both sets of tables in its output directory. All metrics use
area=all and maxDets=100. Means and sample SD (ddof=1) are computed before
rounding; metrics are percentages and paired differences are percentage points.
These are descriptive comparisons, not component-level causal attribution.
