# Efficiency reproduction

Authoritative results are under `publication/`:

- `COMPLEXITY_TABLE.csv`: ten configurations, deployed Params and detector GFLOPs.
- `LATENCY_TABLE.csv`: DEIM, DEIM+GCA and complete HiLoR-DETR, measured together
  using balanced alternating passes in one process.
- `TRAINING_COST_TABLE.csv`: DEIM and DEIM+GCA, two complete 102-epoch runs.

## Complexity

Input is 1×3×640×640; calflops 0.3.2, with two FLOPs per MAC. Params count the actual deployed graph,
including the candidate-value MLP and four fitted mapping scalars when enabled.
Retired fused RepVGG branches and training-only geometry heads are excluded.
GFLOPs cover detector forward only, not postprocessing. Measurement records are
under `outputs/complexity/` for all ten configurations. Each measurement names
its selected seed-0 checkpoint and deployment conversion.

RT-DETR-R18 and its complete HiLoR variant use their selected seed-0 checkpoints
from the second-host package, with strict EMA loading. Detector complexity is
shared by GCA and full HiLoR; only full HiLoR adds CCVR deployment parameters.

## Main-host latency

RTX 4090 (24 GB), Xeon Gold 6430, NVIDIA driver 560.35.03, PyTorch 2.5.1+cu124
(CUDA runtime 12.4), batch 1, 237 real Test images, 50 warmups per configuration,
three balanced alternating passes. Detector forward and complete postprocessing
use FP32 tensors with autocast disabled. As in the main-host AP environment,
TF32 is disabled for matrix multiplication and enabled for cuDNN convolutions.
cuDNN benchmarking is enabled for timing; both PyTorch thread pools use one thread.
Timed scope includes detector forward, required FP32 conversion, native Top-300
or full CCVR and box scaling. It excludes IO, decoding, preprocessing, H2D and
COCO evaluation. Report postprocessing and total latency, not a SOTA-wide FPS claim.

## Main-host training cost

RTX 4090, seed 0, batch 8, FP32, EMA, 102 complete train_one_epoch calls. Timing
includes loading/H2D/forward/loss/backward/optimizer/EMA/logging and excludes
validation and checkpoint saving. Peak is training allocated CUDA memory, not
reserved or inference memory. Full HiLoR reuses GCA detector training; later
CCVR fitting is separate.

## Commands

Run from `efficiency/`, using a fresh external working directory for each explicit stage.
The paired-latency stage uses the main-host environment.

The `latency` stage produces the paired comparison. Any individual timings from
the `inference` stage are diagnostics and are not the publication latency table.

```bash
python -B reproduce_efficiency.py --stage check --workdir /your/new/efficiency_check
python -B reproduce_efficiency.py --stage smoke --workdir /your/new/efficiency_smoke
python -B reproduce_efficiency.py --stage complexity --workdir /your/new/efficiency_complexity
python -B reproduce_efficiency.py --stage latency --workdir /your/new/efficiency_latency
python -B reproduce_efficiency.py --stage training --workdir /your/new/efficiency_training
```

Set `EFFICIENCY_ENV_ROOT` when the matching environments are installed elsewhere
(`deim_cvpr2025/bin/python` and `rtdetr_sota/bin/python`). The profiler is bundled
under `vendor/`. CPU checks are not GPU timing. Measurement scripts and generated
outputs run in the new working copy, leaving the released assets unchanged.
For deeply nested paths, set `EFFICIENCY_TMP_ROOT` to a short temporary directory,
for example `/tmp/hilor_eff`, before running these commands.
