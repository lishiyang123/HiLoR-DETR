# Nine-model SOTA comparison package

This package contains the protocol, source/weights, Train/Val evidence, frozen-Test outputs,
manuscript tables for the following comparison set:

1. YOLOv8-L (official single checkpoint)
2. YOLO11-L (official single checkpoint)
3. RT-DETR-R18 (three seeds)
4. D-FINE-M (three seeds)
5. YOLOv10-M (three seeds)
6. YOLOv12-M turbo (three seeds)
7. YOLOv9-C (three seeds)
8. B0, HGNetv2-M (three seeds)
9. Ours, HGNetv2-M (three seeds)

Final Test metrics use the frozen CDW1630 Test annotation and
`faster-coco-eval==1.7.2`. The release-root `evaluate_test_predictions.py`
replays saved predictions without new inference. Training/Val environment
records retain their original labels. YOLO accuracy inference uses an actual batch of 8. Native model
inference/post-processing is retained where model families differ. Unified reevaluation does
not mean that every architecture has identical native post-processing.

YOLOv8-L and YOLO11-L are official single checkpoints and therefore have no three-seed
standard deviation. The other trained comparison entries report seeds 0/1/2. Checkpoints and
protocols were selected without Test metrics; Test is used only after freezing.

Final paper tables are under `07_PAPER_TABLES/SOTA_NINE_MODEL_PAPER_TABLES_v1/`. Directory
numbers are stable provenance identifiers, not model ranking or a promise of a continuous
sequence.

Run the release-root `run_verify.sh` before using any result. Model-specific reproduction
instructions and environment receipts live inside the corresponding model directory.
`CONFIGURATION_GUIDE.md` centralizes the recorded budgets, initialization,
Val-selection rules and executable training/configuration-planning commands.

Native RT-DETR final Test inference uses the release-root RT entry with deterministic GPU execution. Current three-seed statistics are rebuilt from its saved final predictions and evaluation arrays; use the release quick start. RT Native/GCCS Val selection and Test both use faster-coco-eval 1.7.2.

## Published trained checkpoints

YOLOv9-C, YOLOv10-M and YOLOv12-M use the per-seed
`weights/best_val_ap.pt` selected by Val AP. Native RT-DETR uses
`runs/seedN/best_val.pth`; D-FINE uses the checkpoint identified in
its formal Val-selection and Test-evaluation records. Use these same
selected weights for paper reproduction. Training configurations,
initialization weights and original run/selection records are included.
