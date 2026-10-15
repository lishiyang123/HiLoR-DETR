# SOTA benchmark reproducibility package

This reviewer-facing package contains canonical model assets, formal Train/Val records, frozen-Test predictions and metrics, and paper-ready tables.

All eight reported models use frozen-Test batch size 8. YOLOv8-L and YOLO11-L are official single checkpoints; the other reported trained methods use their validation-selected checkpoints. All accuracy statistics are evaluated with `faster-coco-eval==1.7.2`. Test data are never used for fitting, model selection, or hyperparameter selection.
