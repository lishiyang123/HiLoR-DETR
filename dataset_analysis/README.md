# CDW1630 Dataset Characteristics Reproduction Package v1

This is a conservative, annotation-only evidence package. It reproduces the
basic descriptive statistics of the final CDW1630 Train/Val/Test splits and
binds every input to `main_host` with SHA-256.

Contents:

- `inputs/annotations/`: local copies of the three final COCO JSON files;
- `tables/`: canonical CSV tables;
- `results/`: machine-readable and paper-facing summaries;
- `protocol/`: frozen scope and interpretation boundary;
- `manifests/`: input bindings and package checksum manifest;
- `source/`: the exact analysis program.

The original images are not duplicated. Their existence is verified against
the formal dataset tree. No pixels, models, checkpoints, predictions, GPU,
training, or inference are used. The package reproduces numerical statistics without generating graphical layouts.

<!-- density-overlap-child:start -->
## Conditional performance analysis

`conditional_performance/DENSITY_OVERLAP_STRATIFIED_EVAL_v1/` is a separate post-hoc analysis of B0 versus
the Full model under Train-frozen low/high density and overlap conditions. It
does not modify or broaden the parent package's conservative descriptive
statistics protocol. The child retains three paired detector seeds, exact
full-Test replay gates, raw predictions, whole-image group definitions, and
CPU-reproducible COCO results.
<!-- density-overlap-child:end -->

This release retains numerical analysis only; generated plots and plotting functions are excluded.
