# CDW1630 main-host reproduction

The shared DEIM/HiLoR implementation for CDW1630 and BTC is in
`source/FINAL_CODE/`. Its README documents frozen-checkpoint evaluation, detector
training, Train/Val fitting tools and the profile names. All numerical references
are bound to the supplied checkpoints and fitted CCVR assets.

From the aggregate release root:

```bash
python -u -B main_host/source/FINAL_CODE/reproduce.py \
  --release-root . --profile cdw_full_seed0_test \
  --output /path/to/new/cdw_full_seed0 --compare-reference
```

Use `cdw_deim_seedN_test`, `cdw_gccs_seedN_test`, `cdw_qccr_seedN_test` or
`cdw_full_seedN_test`, with N = 0, 1 or 2. Evaluation uses FP32 EMA weights,
batch size 8 and faster-coco-eval 1.7.2. New outputs must be outside the release.

`results/`, `paper_tables/`, `checkpoints/`, `datasets/`, `protocols/` and
`training_records/` retain the formal experiment assets. Original completed-run
records describe their original runs; use the source profiles for current execution.
`tools/verify_eval.py` extracts and checks all twelve COCO metrics and strict-IoU AP.
The class-wise analysis subpackage provides a CPU-only extraction step;
no forward pass is needed for saved eval tensors.

Verify package contents with `python -B verify_release.py verify` from the
aggregate release root. Test is not used for parameter fitting or selection.
