# Functional module mapping

The method is organized as a continuous three-stage system:

1. **UFD-LCCT (training side).** UFD provides difficulty-aware auxiliary geometry supervision. LCCT is independently controlled and schedules residual-query supervision from localization gain relative to the native matched query. Both are jointly enabled in the released detector, and neither changes deployment-time inference cost.
2. **QCCR (inference calibration).** QCCR recalibrates confidence using checkpoint-specific coefficients fitted on Val under the globally fixed regularization value `lambda=1e-3`.
3. **UMR (inference refinement).** The corrected MLP UMR models overlapping same-class candidate relations. Its global tuple is fixed to `pw=4`, `tau=0`, `beta=2`, `topk=300`, with head-training seed 0; its weights are fitted on Train for the selected detector checkpoint.

The rows report a cumulative stage sequence. UFD and LCCT are not split because the released evidence does not include independent UFD-only and LCCT-only runs—not because one consumes the other's signal. The table therefore supports only the joint training-stage increment.
