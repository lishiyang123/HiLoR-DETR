# Value-residual sensitivity

The study varies tau and beta_v on CDW1630 Val with the GCCS detector fixed
at seed 0 and delta=0.02. The mapping regularization is fixed at 0.001.

- tau: 0, 0.1, 0.2, 0.3, 0.4.
- beta_v: 0, 0.25, 0.5, 1, 2.
- Three value-head fitting seeds: 0, 1, 2; these are not three detector seeds.
- Head fitting uses the 1042-image Train split with a seed-dependent 90/10
  internal holdout. The selected epoch minimizes holdout QFL. Positive weight
  is 4; AdamW uses lr=0.001, weight decay=0.0001, batch size=16,
  at most 50 epochs, and patience=8.
- Evaluation uses all 237 Val images, FP32 CPU postprocessing, Top-300, and
  faster-coco-eval 1.7.2. No Test inputs are used by this study.

The final setting is (tau, beta_v)=(0, 2). It has the highest mean Val AP
among the 25 tested combinations: 76.77346034858611%, with sample standard
deviation 0.012476632264611784 percentage points across the three heads.
The gain over mapping-only (beta_v=0) is 0.15437728154984454 AP points.
This is a comparison within the tested grid, not a claim of a global optimum.

## Files

All paths below are relative to `sensitivity/tau_beta/`.

- `run.py`: replay the grid, optionally refitting the heads.
- `settings.json`: grid, fixed settings and relative input locations.
- `inputs/`: Train/Val captures and the detector-specific mapping.
- `heads/`: the three value heads and their fitting records.
- `results/per_head.csv`: all 75 head/setting results.
- `results/mean.csv`: 25 means, sample standard deviations and baseline gains.
- `results/baseline.json`, `results/summary.json`: reference and summary.

Metric columns use fractions (0 to 1); multiply by 100 for AP percentages.
Columns ending in `_sd` contain sample standard deviations in the same units.
Columns ending in `_pp` already use percentage points. The inherited diagnostic
guards and weighted score are also reported, but the maximum reported above is
the maximum mean AP. All grid points, including inferior points, are retained.

## Reproduce on CPU

Use the main-host environment, including `faster-coco-eval==1.7.2`.
From the release root, write to a new directory outside the release:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
python -B sensitivity/tau_beta/run.py --output /path/to/new/tau_beta
```

This reuses the released heads and Val capture; it does not need images or a GPU.
To also repeat head fitting from the saved Train capture:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
python -B sensitivity/tau_beta/run.py --refit-heads --output /path/to/new/tau_beta_fit
```

For layered downloads, the script is in code, captures in features, heads in
weights, and numerical results in results. Extract them into one release root.
No detector retraining or new detector capture is performed by these commands.
