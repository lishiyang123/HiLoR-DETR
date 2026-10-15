# RT-DETR-R18 experiment protocol

| Setting | Value |
|---|---|
| Dataset | CDW1630; Train/Val/Test = 1144/249/237 images |
| Initialization | Official RT-DETR-R18 COCO checkpoint supplied in the SOTA package |
| Detector seeds | 0, 1, 2 |
| Training | 102 epochs; FP32; batch 4; EMA |
| Evaluation batch | 8 for Val and Test |
| Optimizer | AdamW; lr 1e-4; backbone lr 1e-5; weight decay 1e-4; betas (0.9,0.999) |
| Gradient clipping | 0.1 |
| EMA | decay 0.9999; warmup 2000 |
| LR schedule | MultiStepLR, milestones=[1000], gamma=0.1 |
| Checkpoint selection | Maximum Val AP; earliest exact tie |
| Evaluator | faster-coco-eval 1.7.2; standard COCO bbox; maxDets=[1,10,100] |
| GCCS host adaptation | alpha_u_max=0; lambda_ufd_max=0.75; e0=20; ramp=5 |

The executable training profiles preserve RT-DETR's augmentation and multiscale
schedule. Native and GCCS share those settings. GCCS adds residual geometric
matching, localization-gain gating, VFL-compatible role correction and auxiliary
geometry supervision. RT-DETR does not supply the native boundary distribution
used by the primary host, so alpha_u_max=0 bypasses all pred_corners access.
The training-only auxiliary head is removed for deployment.

Complete HiLoR-DETR reuses the selected GCCS detector, then independently fits
CCVR for each detector seed. It is not a third detector-training configuration.
Test is used only after checkpoint selection and fixed Train/Val fitting.

Test inference uses FP32/EMA, deterministic PyTorch/cuDNN algorithms,
cuDNN benchmark=False, CUBLAS_WORKSPACE_CONFIG=:4096:8 and CPU/OpenMP threads=1.
Saved predictions and evaluation arrays are the numerical sources for tables.
Report means and sample standard deviations (denominator n-1) across three seeds.

Params use the deployed inference graph, including the fitted value head and
four mapping scalars for the complete method. GFLOPs use detector forward only
at 1×3×640×640 with calflops 0.3.2; postprocessing is excluded. The complete
method and GCCS have the same deployed detector graph and box coordinates.

Source and data identities used by the implementation are retained under
`provenance/`. Published run selection and outputs are identified by
`CURRENT_RESULT_MANIFEST.json`, not by upstream source-snapshot checkpoint lists.
