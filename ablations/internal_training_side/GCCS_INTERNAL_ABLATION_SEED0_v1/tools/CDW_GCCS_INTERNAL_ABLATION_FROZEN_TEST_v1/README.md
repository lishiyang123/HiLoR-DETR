# CDW1630 GCCS internal ablation frozen Test

This package evaluates exactly the three newly trained seed-0 rows selected on Val: `UFD w/o difficulty prior`, `UFD-only`, and `LCCT-only`. It reuses the already formal B0 and GCCS native Test evaluations after exact SHA/metric replay, producing one five-row internal training-side ablation table.

The protocol is fixed before inference: CDW1630 Test (237 images), physical batch 8, FP32, 640x640, native detector output, QCCR off, and UMR off. Test is reporting-only. It does not select a checkpoint, fit a parameter, choose a row, or change the final release.

Upload and extract the source zip at `/root/autodl-tmp/BTC_DEIM_project`, then run:

```text
bash CDW_GCCS_INTERNAL_ABLATION_FROZEN_TEST_v1/run_server.sh --preflight
bash CDW_GCCS_INTERNAL_ABLATION_FROZEN_TEST_v1/run_server.sh --start
bash CDW_GCCS_INTERNAL_ABLATION_FROZEN_TEST_v1/run_server.sh --status
```

If the server stops, rerun `--start`; completed rows are verified and skipped. After the queue reports `COMPLETE`:

```text
bash CDW_GCCS_INTERNAL_ABLATION_FROZEN_TEST_v1/run_server.sh --verify
bash CDW_GCCS_INTERNAL_ABLATION_FROZEN_TEST_v1/run_server.sh --summary
```

The package writes only `CDW_GCCS_INTERNAL_ABLATION_FROZEN_TEST_STATE_v1`. Training outputs and `HiLoR-DETR` remain read-only.
