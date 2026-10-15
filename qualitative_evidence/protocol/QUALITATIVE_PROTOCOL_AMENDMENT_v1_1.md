# Qualitative protocol amendment v1.1

The frozen v1 analysis used image-level F1 at a common score threshold of 0.30
to rank success and failure cases. That diagnostic revealed a comparability
problem: QCCR and UMR intentionally alter score distributions, so an absolute
score cutoff can dominate the per-image F1 difference even when the official
threshold-integrated COCO evaluation is unchanged.

Revision v1.1 changes reporting analysis only. It does not retrain a detector,
refit QCCR or UMR, rerun prediction, alter an official metric, or use Test for
parameter/model selection. Existing v1 output is retained as superseded
diagnostic history and is excluded from the paper figure.

Formal case ranking now uses a class-aware, 101-point interpolated per-image AP
computed from the top 100 detections without an absolute score cutoff. A case
can be labelled a balanced localization gain only when image mAP@[.50:.95] and
AP75 improve while AP50 does not decrease. No ineligible case is used to fill a
shortfall. Two lowest-change cases are retained for disclosed failure analysis.

The score threshold of 0.30 remains a rendering-only setting for readable box
overlays and never affects formal v1.1 case selection.
