# Parameter Sensitivity Reproduction Package v1

This completed package separates training-side and inference-side sensitivity evidence.

- `01_TRAINING_SIDE`: five-point LCCT-delta study using seed 0 and Val reporting.
- `02_INFERENCE_SIDE/QCCR_LAMBDA`: four-candidate OOF-Val QCCR regularization study.
- `02_INFERENCE_SIDE/UMR_TAU_BETA`: corrected-semantics UMR tau-by-beta study with `pw=4` and `topk=300` fixed.
- `03_COMBINED_SUMMARY`: final numerical tables and robustness statement.
- `04_FIGURE_HANDOFF`: SHA-bound inputs for the final sensitivity figure.

For QCCR, `lambda_reg=1e-3` is the predeclared frozen protocol value. The OOF-Val table is a diagnostic sensitivity analysis: no candidate satisfies every guard. The retained legacy token `OOF_VAL_GO` means that the predeclared fixed protocol is activated; it does not mean that the candidate qualified under all guards.

Test data were not used for parameter selection. This reviewer package reports completed evidence and does not execute detector training, fitting, inference, evaluation, or figure generation during verification.

Figure handoff templates are excluded from this release. Unrounded plotting numbers are collected in `../numerical_evidence/evidence/parameter_sensitivity/`; experiment configurations and metric records remain here.
