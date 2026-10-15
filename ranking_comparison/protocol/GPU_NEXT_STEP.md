# GPU next step: bound seed-0 Test comparison

Status: completed successfully on 2026-10-07.  The commands are retained as
the exact reproduction protocol for the frozen Test comparison.

Run these commands from:

`/root/autodl-tmp/BTC_DEIM_project/RELEASE_RENAME_TRIAL_20261015/HiLoR-DETR/ranking_comparison`

They write only inside this isolated study directory.  The formal package is
read-only.

## 1. Capture only the missing Test stream

```bash
PYTHONPATH=source /root/autodl-tmp/envs/deim_cvpr2025/bin/python -B \
  source/capture_val_test.py \
  --branch-root /root/autodl-tmp/BTC_DEIM_project/RELEASE_RENAME_TRIAL_20261015/HiLoR-DETR/main_host/source/OPTIMIZED_REPRO_SOURCE \
  --checkpoint /root/autodl-tmp/BTC_DEIM_project/RELEASE_RENAME_TRIAL_20261015/HiLoR-DETR/main_host/checkpoints/CDW1630/Ours/seed0/best_stg2.pth \
  --val-config /root/autodl-tmp/BTC_DEIM_project/RELEASE_RENAME_TRIAL_20261015/HiLoR-DETR/main_host/protocols/materialized_configs/CDW1630/cdw_capture_val.yml \
  --test-config /root/autodl-tmp/BTC_DEIM_project/RELEASE_RENAME_TRIAL_20261015/HiLoR-DETR/main_host/protocols/materialized_configs/CDW1630/cdw_native_test.yml \
  --output-dir state/captures \
  --expected-checkpoint-sha 7774e65d01e0bfe1774022095dc29c6f60bd75b9f535fee3eb850de0452d2885 \
  --num-classes 5 \
  --capture-prefix gccs_seed0 \
  --splits test
```

## 2. Run the eight disclosed replay paths

```bash
PYTHONPATH=source /root/autodl-tmp/envs/deim_cvpr2025/bin/python -B \
  source/run_frozen_comparison.py \
  --capture state/captures/gccs_seed0_test_capture.pt \
  --annotation /root/autodl-tmp/BTC_DEIM_project/RELEASE_RENAME_TRIAL_20261015/HiLoR-DETR/main_host/datasets/CDW1630/annotations/instances_test2017.json \
  --split test \
  --output-root results/test/seed0_comparison \
  --linear-iou-threshold 0.95 \
  --gaussian-sigma 2.0 \
  --gilg-reference-params state/fitted/gilg_original_seed0.json \
  --gilg-complete-pool-params state/fitted/gilg_complete_pool_seed0.json \
  --formal-source /root/autodl-tmp/BTC_DEIM_project/RELEASE_RENAME_TRIAL_20261015/HiLoR-DETR/main_host/source/OPTIMIZED_REPRO_SOURCE \
  --qccr-params /root/autodl-tmp/BTC_DEIM_project/RELEASE_RENAME_TRIAL_20261015/HiLoR-DETR/main_host/checkpoints/CDW1630/Ours/seed0/calibration/qccr_params.json \
  --umr-checkpoint /root/autodl-tmp/BTC_DEIM_project/RELEASE_RENAME_TRIAL_20261015/HiLoR-DETR/main_host/checkpoints/CDW1630/Ours/seed0/calibration/umr_mlp_head_seed0_pw4.pt \
  --umr-metadata /root/autodl-tmp/BTC_DEIM_project/RELEASE_RENAME_TRIAL_20261015/HiLoR-DETR/main_host/checkpoints/CDW1630/Ours/seed0/calibration/umr_mlp_head_seed0_pw4.metadata.json \
  --formal-replay-device cuda \
  --iou-reg-checkpoint state/fitted/iou_regression/iou_regressor_seed123.pt \
  --iou-reg-metadata state/fitted/iou_regression/iou_regressor_seed123.metadata.json \
  --iou-reg-device cuda
```

This preserves every output.  Test comparison is unrestricted by project
decision; the commands do not delete or hide an unfavorable result.
