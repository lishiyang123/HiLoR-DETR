# Nine-model SOTA comparison package

This package contains the protocol, source/weights, Train/Val evidence, frozen-Test outputs,
manuscript tables for the following comparison set:

1. YOLOv8-L (three seeds)
2. YOLO11-L (three seeds)
3. RT-DETR-R18 (three seeds)
4. D-FINE-M (three seeds)
5. YOLOv10-M (three seeds)
6. YOLOv12-M turbo (three seeds)
7. YOLOv9-C (three seeds)
8. B0, HGNetv2-M (three seeds)
9. Ours, HGNetv2-M (three seeds)

The complete RT-DETR-R18 + HiLoR configuration is provided separately in
`../second_host/`. Combining it with these nine rows gives the ten configurations
in `../efficiency/publication/COMPLEXITY_TABLE.csv`; the directory name of the
nine-row accuracy table does not include that additional second-host result.

Final Test metrics use the frozen CDW1630 Test annotation and
`faster-coco-eval==1.7.2`. The release-root `evaluate_test_predictions.py`
replays saved predictions without new inference. Training/Val environment
records retain their original labels. YOLO accuracy inference uses an actual batch of 8. Native model
inference/post-processing is retained where model families differ. Unified reevaluation does
not mean that every architecture has identical native post-processing.

All comparison entries report seeds 0/1/2. Checkpoints and
protocols were selected without Test metrics; Test is used only after freezing.

Model-specific results live with their corresponding model inputs. Main-host
tables are under `../main_host/paper_tables/three_seed/`; RT Native/GCA/full
tables are under `../second_host/paper_tables/`. Directory numbers identify model
asset groups, not a model ranking.

Start with the release-root `README.md`. The full-download `run_verify.sh`
check is optional; partial downloads need only the inputs for their selected task.
Model-specific reproduction instructions and environment records live in the corresponding model directory.
`CONFIGURATION_GUIDE.md` centralizes the recorded budgets, initialization,
Val-selection rules and executable training/configuration-planning commands.

Native RT-DETR final Test inference uses the release-root RT entry with deterministic GPU execution. Current three-seed statistics are rebuilt from its saved final predictions and evaluation arrays; use the release quick start. RT Native/GCA Val selection and Test both use faster-coco-eval 1.7.2.

## Published trained checkpoints

YOLOv8-L, YOLO11-L, YOLOv9-C, YOLOv10-M and YOLOv12-M use the per-seed
`weights/best_val_ap.pt` selected by Val AP. Native RT-DETR uses
`runs/seedN/best_val.pth`; D-FINE uses the checkpoint identified in
its formal Val-selection and Test-evaluation records. Use these same
selected weights for paper reproduction. Training configurations,
initialization links and original run/selection records are included.
