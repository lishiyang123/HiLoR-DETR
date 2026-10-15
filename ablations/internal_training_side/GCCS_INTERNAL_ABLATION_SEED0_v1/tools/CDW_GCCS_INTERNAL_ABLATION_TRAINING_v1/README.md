# CDW1630 GCCS internal ablation training

Upload the zip to `/root/autodl-tmp/BTC_DEIM_project`, extract it there, and keep the package directory name unchanged.

This package runs three new seed-0 Train/Val experiments: `UFD w/o difficulty prior`, `UFD-only`, and `LCCT-only`. Existing B0 and GCCS rows are reused only after their checkpoint and log hashes match the final release. All rows use 102 epochs, physical batch 8, FP32, the same optimizer/augmentation/pretrained checkpoint, and native detector output with QCCR/UMR disabled.

The no-difficulty control keeps UFD intact and sets only `ufd_alpha_h=0`, which removes the object-level difficulty term from token-conflict priority and auxiliary loss weighting.

Recommended sequence:

```text
bash CDW_GCCS_INTERNAL_ABLATION_TRAINING_v1/run_server.sh --preflight
bash CDW_GCCS_INTERNAL_ABLATION_TRAINING_v1/run_server.sh --prepare
bash CDW_GCCS_INTERNAL_ABLATION_TRAINING_v1/run_server.sh --smoke
bash CDW_GCCS_INTERNAL_ABLATION_TRAINING_v1/run_server.sh --start
bash CDW_GCCS_INTERNAL_ABLATION_TRAINING_v1/run_server.sh --status
```

If the server stops, rerun `--start`. The package resumes a partial variant from its bounded `last.pth`. After the queue completes:

```text
bash CDW_GCCS_INTERNAL_ABLATION_TRAINING_v1/run_server.sh --verify
bash CDW_GCCS_INTERNAL_ABLATION_TRAINING_v1/run_server.sh --summary
```

This package never reads the Test split and never modifies the final release.

