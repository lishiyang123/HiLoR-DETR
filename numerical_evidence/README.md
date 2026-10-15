# Paper numerical evidence reproduction package

This package replaces the previous generated-figure package. It contains the
unrounded numerical evidence for the five supplied figures, not PNG/PDF/SVG
exports, graphical layouts or plotting/rendering programs.

| Supplied figure | Evidence folder | Contents |
|---|---|---|
| 1: three-panel sensitivity | `evidence/parameter_sensitivity/` | delta five points; four mapping regularizers; tau x beta grid and source seeds |
| 2: qualitative 3x3 | `evidence/qualitative_cases/` | images 107/160/158, GT/prediction boxes, classes, scores, Top-300 ranks, IoUs and case metadata |
| 3: AP-IoU curve | `evidence/ap_iou_curve/` | all thresholds, three seeds, means and sample SD of paired deltas |
| 4: gain threshold | shared image-1 delta CSV | five threshold values and their Val metrics |
| 5: introductory comparison | `evidence/introduction_comparison/` | image 34 R1/R2, actual GT/prediction coordinates, scores and IoUs |

`NUMERICAL_EVIDENCE_INDEX.json` is the navigation entry. `SOURCE_BINDINGS.json`
records each original source path, byte count and SHA. All original JSON/CSV
contents are retained without changing numbers. Historical paths and display
policy fields inside source evidence are provenance, not commands to redraw.
Use unrounded numbers as authoritative; software figure labels can be rounded.

The GCCS assignment capture is also retained. The complete oracle geometry
subpackage remains under `evidence/ap_iou_curve/ORACLE_GEOMETRIC_COVERAGE_REPRO_v1/`,
including six full-query captures, per-GT/per-seed results and calculation code.
Its depth relative to the release root is unchanged for its existing launcher.

Data images, annotations, official predictions and evaluator arrays remain in
their original experiment packages. They are not replaced by software artwork.
For recomputation use those numerical analysis/reproduction tools; this package
does not automatically reconstruct any graphical layout. No training, inference
or parameter selection was performed when organizing these files.
