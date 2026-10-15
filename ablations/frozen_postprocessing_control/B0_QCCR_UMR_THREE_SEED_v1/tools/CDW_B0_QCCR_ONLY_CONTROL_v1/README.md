# B0 + QCCR three-seed frozen-postprocessing control

This bounded add-on evaluates QCCR alone on the three already frozen B0
checkpoints. It reuses the checkpoint-specific QCCR artifacts produced by the
completed `CDW_B0_QCCR_UMR_CONTROL_v1` experiment. UMR is explicitly disabled.

No detector is trained and no calibration component is refitted. The existing
B0 and B0+QCCR+UMR rows are reused by SHA-256 binding; only the missing
B0+QCCR formal Test row is newly evaluated.

```bash
bash CDW_B0_QCCR_ONLY_CONTROL_v1/run_server.sh --preflight
bash CDW_B0_QCCR_ONLY_CONTROL_v1/run_server.sh --prepare-test
bash CDW_B0_QCCR_ONLY_CONTROL_v1/run_server.sh --start-test
bash CDW_B0_QCCR_ONLY_CONTROL_v1/run_server.sh --test-status
bash CDW_B0_QCCR_ONLY_CONTROL_v1/run_server.sh --verify
bash CDW_B0_QCCR_ONLY_CONTROL_v1/run_server.sh --summary
```
