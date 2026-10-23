# CCVR candidate-value fitting sources

CDW1630 Train/Val contain 1,042/237 images. Detector seeds are 0/1/2;
the value-head initialization seed is 0. This is a Val-only comparison.

| Fitting source | AP | AP50 | AP75 | AR100 | Paired AP increment (pp) |
|---|---:|---:|---:|---:|---:|
| Mapping only | 76.527 | 96.590 | 87.702 | 86.780 | -- |
| Train captures | 76.699 | 96.640 | 87.865 | 86.780 | +0.172 +/- 0.071 |
| Val-OOF | 76.637 | 96.652 | 87.772 | 86.780 | +0.110 +/- 0.054 |
| Matched-size Train | 76.624 | 96.672 | 87.766 | 86.780 | +0.097 +/- 0.034 |

AP/AR are three-seed means in percent; the increment's +/- is the sample SD
of the three paired differences, not a confidence interval. JSON/CSV retain
unrounded metrics as fractions and increments as percentage points.

The five image-level Val folds contain 48/48/47/47/47 images. For each target
fold, the mapping and Val-OOF head are fitted on the other four folds.
All four rows use that fold-specific mapping during evaluation. The primary
Train row reuses its detector-specific formal Train-fitted head.
The matched Train control uses 189/190 images, equal to the four-fold Val count;
the internal 90/10 split gives 170/171 fitting images and 19 early-stopping images.
Pooled held-out detections are evaluated once; fold APs are not averaged.

Fixed settings: lambda_map=0.001; positive weight=4; tau=0; beta=2; Top-K=300;
AdamW learning rate=0.001, weight decay=0.0001, batch=16 images, maximum=50 epochs,
patience=8. The target fold is not used for early stopping.
The primary Train head uses more fitting images, so its larger average gain
does not isolate fitting source from sample size.

## CPU commands

From the release root, use a new external output directory:

```bash
python -B main_host/robustness/ccvr_fitting_sources/run.py --stage tables --output /new/external/fitting_tables
python -B main_host/robustness/ccvr_fitting_sources/run.py --stage metrics --output /new/external/fitting_metrics
python -B main_host/robustness/ccvr_fitting_sources/run.py --stage fit --output /new/external/fitting_heads
```

`tables` aggregates the saved seed results. `metrics` evaluates the twelve saved
prediction files with faster-coco-eval 1.7.2. Both need code + results only.
`fit` additionally needs the shared GCA Train/Val captures and the three formal
Train-fitted heads; it does not need detector weights or dataset images.
All paths are relative to the release in `inputs.json`; no author workspace is required.
Use `--release /path/to/release` when needed. No command trains a detector or reads Test data.

`seed0/seed1/seed2` contain the fold heads, fitting histories, pooled predictions
and per-seed metrics. `results/` contains the three-seed summary.
