# CDW1630 dataset-characteristics protocol

Status: **frozen before analysis**. Profile: **conservative core only**.

The analysis reads the final COCO annotation JSON files for the frozen Train,
Val, and Test splits and reports only four neutral characteristics: split
size, category distribution, COCO bounding-box scale, and annotated-object
count per image. Image paths are checked for completeness, but image pixels
are never opened.

The package intentionally excludes annotation-box overlap, IoMin, class
co-occurrence, model outputs, significance testing, and causal interpretation.
These exclusions prevent descriptive statistics from being presented as
post-hoc justification for the method design.

Small, medium, and large follow the standard COCO bounding-box area cutoffs of
`32^2` and `96^2` pixels. Frozen Test annotations are used only for dataset
description and never for model, checkpoint, or parameter selection.
