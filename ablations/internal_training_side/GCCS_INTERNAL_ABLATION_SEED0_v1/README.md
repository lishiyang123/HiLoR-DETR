# GCCS training-side ablations

The seed-0 comparison includes the native detector, geometric supervision
without the difficulty prior, geometric supervision only, classification-role
adjustment only, and GCCS. The three independently trained internal variants
retain their Val-selected checkpoints and formal Test evaluations. Native and
GCCS results are reused by asset binding. CCVR is disabled for this comparison.

From the aggregate release root:

```bash
python -u -B main_host/source/FINAL_CODE/reproduce.py \
  --release-root . --profile cdw_ufd_only_seed0_test \
  --output /path/to/new/ufd_only --compare-reference
```

The three evaluation profiles are `cdw_lcct_only_seed0_test`,
`cdw_ufd_only_seed0_test` and `cdw_ufd_without_difficulty_seed0_test`.
Their training profiles replace `_seed0_test` with `_train`; these use each
variant's original recorded Train/Val configuration and the official COCO
initialization. The local training and evaluation `manager.py` launchers accept
the shared source's `--profile`, `--output`, `--seed` (training) and
`--preflight-only` arguments. Completed-run settings remain in `configs/` and
the frozen evidence directories. Test is reporting-only.
