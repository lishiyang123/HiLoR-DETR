# CDW calibration: fixed protocol, checkpoint-specific fits

The runtime and trainer both import the same MLP `engine.deim.candidate_value.UMRHead`.
Do not use a head or QCCR coefficients fitted for a different detector checkpoint.

This release retains the existing protocol: QCCR regularization 0.001; UMR
pw4/tau0/beta2/topk300, head seed0. QCCR coefficients are fitted on Val, and the
UMR head is trained on Train, using the existing Train-internal holdout. The
detector checkpoint is selected on Val. The MLP architecture was compared with
Attention on Val before its final Test evaluation.

OOF and UMR grid reports are retained as diagnostics; they do not replace this
fixed tuple with a newly selected tuple for each detector. The original fixed-rule
adapters are under `../fixed_protocol/`. They preserve the original diagnostic
decisions in their output; a fixed-policy freeze is not evidence that all optional
OOF safety gates passed. Test evaluation is a separate operation, not a fitting
input. These statements describe this release workflow, not a retrospective
claim that every earlier exploratory experiment used the same protocol.

## Refit from a selected detector

Run from the source root. Substitute `CHECKPOINT_SHA256`, dataset and output paths.
Outputs must be new directories. First materialize configs as explained in the
source README, then capture Train/Val:

```bash
python -B tools/calibration/common/capture_train_val.py \
  --branch-root . --checkpoint /path/to/selected_detector.pth \
  --train-config /path/to/configs/cdw_capture_train.yml \
  --val-config /path/to/configs/cdw_capture_val.yml \
  --output-dir /path/to/calibration/capture \
  --expected-checkpoint-sha CHECKPOINT_SHA256 --num-classes 5 --capture-prefix cdw
```

Fit QCCR and retain the original OOF report:

```bash
python -B tools/qccr/build_qccr_targets.py \
  --capture /path/to/calibration/capture/cdw_val_capture.pt \
  --output /path/to/calibration/qccr/targets.pt \
  --summary /path/to/calibration/qccr/target_summary.json
python -B tools/qccr/fit_qccr.py --targets /path/to/calibration/qccr/targets.pt \
  --output /path/to/calibration/qccr/fits.json --seed 0 --folds 5
python -B tools/qccr/evaluate_qccr_oof_val.py \
  --capture /path/to/calibration/capture/cdw_val_capture.pt \
  --baseline-eval /path/to/calibration/capture/cdw_val_baseline_eval.pth \
  --fits /path/to/calibration/qccr/fits.json \
  --annotation /path/to/annotations/instances_val2017.json \
  --output-dir /path/to/calibration/qccr/oof_val
python -B tools/calibration/fixed_protocol/fix_qccr.py \
  --summary /path/to/calibration/qccr/oof_val/oof_summary.json \
  --fits /path/to/calibration/qccr/fits.json \
  --policy tools/calibration/fixed_protocol/PROTOCOL_LOCK.json \
  --output /path/to/calibration/qccr/fixed_summary.json
python -B tools/calibration/common/freeze_qccr.py \
  --oof-summary /path/to/calibration/qccr/fixed_summary.json \
  --fits /path/to/calibration/qccr/fits.json \
  --checkpoint /path/to/selected_detector.pth \
  --output /path/to/calibration/qccr/qccr_params.json \
  --expected-checkpoint-sha CHECKPOINT_SHA256 \
  --dataset-name CDW1630 --source-method UFD_LCCT_CONTINUOUS_HANDOVER
```

Train the MLP with the unchanged UMR training procedure and bind the fixed tuple:

```bash
python -B tools/calibration/common/train_umr_select_val.py \
  --source-root . --checkpoint /path/to/selected_detector.pth \
  --train-capture /path/to/calibration/capture/cdw_train_capture.pt \
  --val-capture /path/to/calibration/capture/cdw_val_capture.pt \
  --qccr-artifact /path/to/calibration/qccr/qccr_params.json \
  --val-annotation /path/to/annotations/instances_val2017.json \
  --output-dir /path/to/calibration/umr_candidates \
  --expected-checkpoint-sha CHECKPOINT_SHA256 --num-classes 5 --dataset-name CDW1630
python -B tools/calibration/fixed_protocol/fix_umr.py \
  --candidate-dir /path/to/calibration/umr_candidates \
  --policy tools/calibration/fixed_protocol/PROTOCOL_LOCK.json \
  --checkpoint-sha CHECKPOINT_SHA256 --output /path/to/calibration/UMR_FIXED.json
```

The retained trainer computes its usual grids; for this fixed protocol, use
`UMR_FIXED.json` and `models/umr_seed0_pw4_best.pt`, not its automatic grid winner.
The supplied source's launcher is deliberately bound to the existing seed0
detector. New detector fits require their own asset binding/configuration and
provenance, rather than overwriting this frozen asset set. No new fit is run by
the source installer itself.
