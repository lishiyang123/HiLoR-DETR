# CDW1630 Final Ablation Reproduction v1

## Current executable source

Use `main_host/source/FINAL_CODE/reproduce.py` from the aggregate
release root. All profiles and bound assets are listed in its `PROFILES.json`.
Internal ablation evaluation profiles are `cdw_lcct_only_seed0_test`,
`cdw_ufd_only_seed0_test` and `cdw_ufd_without_difficulty_seed0_test`; their
training profiles replace `_seed0_test` with `_train`. The five fixed gate
training profiles are `cdw_delta0005_train`, `cdw_delta001_train`,
`cdw_delta002_train`, `cdw_delta004_train` and `cdw_delta008_train`.
The separately fitted baseline CCVR control uses `cdw_deim_ccvr_seedN_test`,
with N = 0, 1 or 2.

The local training/evaluation `manager.py` launchers accept `--profile`,
`--seed` (training only), and `--output` in the same format as the shared
entry. Completed-run configurations and records below describe the supplied
evidence; use the profile entry above for new execution.


This package contains the final seed0 cumulative ablation for the released method. The four rows are B0, UFD-LCCT, UFD-LCCT+QCCR, and UFD-LCCT+QCCR+UMR.

UFD and LCCT are independently configurable training-side mechanisms and are jointly enabled in the released detector. UFD adds difficulty-aware auxiliary geometry supervision; LCCT applies a bounded continuous curriculum using residual-query localization gain relative to the native matched query. The released table is cumulative and contains no UFD-only or LCCT-only row, so it supports the joint training-stage increment but not separate-effect or strict interaction claims.

The four official-Test `eval.pth` files and exact metrics are included locally. Selected detector checkpoints and the complete final source are not duplicated; they are SHA-bound through relative paths to the sibling `main_host`. This package is intended to be distributed inside the complete paper reproduction release together with that main package.

Contents:

- `results/seed0`: four official-Test evaluation files and exact metric records.
- `paper_tables`: full 16-metric table, incremental deltas, compact manuscript table, and JSON summary.
- `assets/calibration/seed0`: final QCCR/UMR inference assets and binding.
- `evidence/training/seed0`: compact B0/Ours training records.
- `configs`: final-source configuration templates and the materialized Ours training configuration when available.
- `protocol`: frozen cumulative-ablation design and module rationale.
- `bindings/seed0`: source/checkpoint/calibration/result SHA relationships.
- `scripts`: package verification source.
- `manifests`: complete SHA-256 manifest.

Detector checkpoints were selected on Val. QCCR and UMR were fitted/selected under the frozen Train/Val protocol. Test was not used for checkpoint, module, architecture, or hyperparameter selection.

Verify from the common release root:

```bash
python -B ablations/scripts/verify_package.py verify --project /path/to/release/root
```

<!-- GCCS_INTERNAL_ABLATION_CHILD_START -->
## Training-side internal ablation

The child package `internal_training_side/GCCS_INTERNAL_ABLATION_SEED0_v1`
contains the seed-0 five-row training-side comparison: B0, UFD without the
difficulty prior, UFD-only, LCCT-only, and GCCS (UFD+LCCT). The three new rows
include their Val-selected checkpoints and formal Test evaluations. B0 and
GCCS are reused by relative-path SHA binding. QCCR and UMR are disabled for
all five rows. Test is reporting-only and was not used to select checkpoints
or tune the variants.
<!-- GCCS_INTERNAL_ABLATION_CHILD_END -->

<!-- B0_QCCR_UMR_CONTROL_START -->
## Frozen-postprocessing controls

The child package `frozen_postprocessing_control/B0_QCCR_UMR_THREE_SEED_v1`
contains the three-seed B0 branch: B0, B0+QCCR, and B0+QCCR+UMR. It also
provides the six-row seed0 paper table by joining this branch to the released
cumulative GCCS ablation. Detector checkpoints and checkpoint-specific QCCR
and UMR assets are fixed before Test reporting.
<!-- B0_QCCR_UMR_CONTROL_END -->
