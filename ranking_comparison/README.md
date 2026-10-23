# Candidate-ranking comparison

CDW1630, GSA detector seed 0, 237 Test images / 1,071 objects. Five external
methods use the same saved 300-query x 5-class candidate pool. The Native GSA
and complete CCVR reference rows are the corresponding main-pipeline outputs,
not separately fitted controls. All methods preserve box coordinates.

Results: `results/comparison.csv` (full precision) and `results/comparison.json`.
Per-method predictions are losslessly compressed as JSON.gz. The table reports
AP, AP50, AP75, AP80, AP85, AP90, AP95 and AR100; AP differences are relative
to Native GSA. This is a fixed-detector-seed comparison, not a three-seed table.

## Settings and fitting

- Common processing: CUDA FP32 box conversion/scaling and native sigmoid;
  final global `torch.topk` over query-class candidates, budget 300, no added
  score threshold or NMS. Soft-NMS and Gilg's reference prefilter are explicit
  method-specific operations. COCO evaluation uses faster-coco-eval 1.7.2,
  with `maxDets=[1,10,100]` for the reported metrics.
- Linear Soft-NMS: Val grid `0.3,0.5,0.7,0.9,0.95,0.99`; selected IoU **0.9**.
- Gaussian Soft-NMS: Val sigma grid `0.3,0.5,0.7,1,1.25,1.5,2,3`; selected **2**.
  Both use highest Val AP, then AP75, then the smaller grid value. Every grid
  result and the selection rule are in `state/selection/soft_nms.json`.
- Gilg: conditionally dependent Beta calibration fitted on Val with one-to-one
  TP labels at IoU 0.5; seed 123, maximum 1,000 optimizer iterations. The
  reference variant removes near-duplicates at IoU 0.99; the complete-pool
  variant does not. Both fitted parameter files include convergence results.
  This is an independent implementation of the reference formulation.
- IoU regression: frozen decoder features; LayerNorm(256) plus a five-output
  linear layer; score = native sigmoid score x predicted IoU. Targets are the
  maximum same-class, non-crowd GT IoU. Fit on Train, select by Val BCE.
  AdamW, learning rate 0.003, weight decay 0.0001, 16 images/batch, 20 epochs,
  patience 5, fitting seed 123. Selected epoch **19**, Val BCE
  **0.08463586322768328**. The head, settings and full fitting history are in
  `state/fitted/iou_regression/`.

## Run

Use the main-host environment (Torch 2.5.1+cu124). From the release root, choose
a new output directory outside it:

```bash
# Recompute all seven rows from saved predictions: CPU, no model weights.
python -B ranking_comparison/run.py --stage metrics --output /new/external/ranking_metrics

# Replay the five external methods with supplied fits: CUDA, no detector forward.
python -B ranking_comparison/run.py --stage replay --output /new/external/ranking_replay

# Repeat Val selection and Train/Val fitting, then evaluate: CUDA, no detector training.
python -B ranking_comparison/run.py --stage all --output /new/external/ranking_fit
```

`settings.json` locates all inputs relative to the release root. Train/Val
captures are shared with `sensitivity/tau_beta/inputs/`, so they are not copied
here. Saved-metric evaluation needs code + results; candidate replay also needs
the small IoU-regression head from weights; fresh fitting additionally needs
the optional Train/Val feature captures. Dataset images are not required.
CUDA is used for common postprocessing to match the main pipeline's FP32
rounding and Top-K behavior; CPU-only saved-prediction evaluation is separate.
These commands do not measure deployment latency.
