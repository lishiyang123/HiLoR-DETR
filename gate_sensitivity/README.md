# Localization-gain threshold sensitivity

This seed-0 Train/Val study uses the fixed grid 0.005, 0.01, 0.02, 0.04 and
0.08. Test is not used. The completed 102-epoch FP32 runs, original settings,
and Val results are retained in `configs/`, `outputs/`, `results/` and the
input/protocol bindings. The 0.02 center reuses the recorded formal run.

For fresh execution, use the shared source from the aggregate release root:

```bash
python -u -B main_host/source/FINAL_CODE/reproduce.py \
  --release-root . --profile cdw_delta0005_train --seed 0 \
  --output /path/to/new/delta0005
```

The five profiles are `cdw_delta0005_train`, `cdw_delta001_train`,
`cdw_delta002_train`, `cdw_delta004_train` and `cdw_delta008_train`.
Each configuration is flattened from that grid point's recorded configuration;
training settings are not inferred from Test outputs. `--preflight-only` checks
assets and paths without writing files or using a GPU. New outputs must stay
outside the release. The local `tools/LCCT_FIVE_POINT_TRAINING_SENSITIVITY_v1/manager.py`
and `run_server.sh` accept the same profile arguments. Use the release root's
`verify_release.py verify` to verify the frozen assets.
