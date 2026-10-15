# CDW1630 three-seed reproduction protocol

The final detector is UFD-LCCT with bounded continuous handover. GGLH-PD is disabled.
Inference uses checkpoint-specific QCCR coefficients and checkpoint-specific MLP UMR weights.
The global hyperparameters are fixed for every detector seed: QCCR lambda=1e-3 and
UMR pw=4, tau=0, beta=2, topk=300, head-training seed=0.

For seed 0, the default materialized CDW1630 configuration is already bound to the
seed-0 assets. For seed 1 or 2, first materialize the CDW1630 templates in a temporary
base directory, then run `tools/reproduction/bind_checkpoint_assets.py` using:

- `checkpoints/CDW1630/Ours/seedN/best_stg2.pth`
- `checkpoints/CDW1630/Ours/seedN/calibration/qccr_params.json`
- `checkpoints/CDW1630/Ours/seedN/calibration/umr_mlp_head_seed0_pw4.pt`
- `checkpoints/CDW1630/Ours/seedN/calibration/umr_mlp_head_seed0_pw4.metadata.json`

The bound directory can then be passed to `tools/reproduction/run.py --action test`
with the matching detector seed and checkpoint. The binding tool verifies all SHA256
relationships and refuses cross-seed assets.

Training uses 102 epochs, FP32 and the same data split/protocol for seeds 0, 1 and 2.
Detector checkpoints are selected by Val AP. QCCR is fitted on Val under the fixed
regularization value; UMR heads are fitted on Train and use the fixed fusion tuple.
Test is used only for final evaluation.
