# Efficiency reproduction

Authoritative results are under `publication/`:

- `COMPLEXITY_TABLE.csv`: ten configurations, deployed Params and detector GFLOPs.
- `LATENCY_TABLE.csv`: DEIM, DEIM+GCCS and complete HiLoR-DETR, measured together
  using balanced alternating passes in one process.
- `TRAINING_COST_TABLE.csv`: DEIM and DEIM+GCCS, two complete 102-epoch runs.
- `FINAL_RESULTS_INDEX.json`: exact result and measurement bindings.

## Complexity

Input is 1×3×640×640; calflops 0.3.2. Params count the actual deployed graph,
including the candidate-value MLP and four fitted mapping scalars when enabled.
Retired fused RepVGG branches and training-only geometry heads are excluded.
GFLOPs cover detector forward only, not postprocessing. Measurement records are
under `outputs/complexity/`; base-model evidence is under `results/`.

RT-DETR-R18 and its complete HiLoR variant use their selected seed-0 checkpoints
from the second-host package, with strict EMA loading. Detector complexity is
shared by GCCS and full HiLoR; only full HiLoR adds CCVR deployment parameters.

## Main-host latency

RTX 4090, batch 1, 237 real Test images, 50 warmups, three balanced alternating
passes. Detector forward uses FP16 autocast; complete postprocessing uses FP32.
Timed scope includes detector forward, required FP32 conversion, native Top-300
or full CCVR and box scaling. It excludes IO, decoding, preprocessing, H2D and
COCO evaluation. Report postprocessing and total latency, not a SOTA-wide FPS claim.

## Main-host training cost

Same GPU, seed 0, batch 8, FP32, EMA, 102 complete train_one_epoch calls. Timing
includes loading/H2D/forward/loss/backward/optimizer/EMA/logging and excludes
validation and checkpoint saving. Peak is training allocated CUDA memory, not
reserved or inference memory. Full HiLoR reuses GCCS detector training; later
CCVR fitting is separate.

## Commands

Use a fresh external working directory for each explicit stage:

```bash
python -B reproduce_efficiency.py --stage check --workdir /your/new/efficiency_check
python -B reproduce_efficiency.py --stage smoke --workdir /your/new/efficiency_smoke
python -B reproduce_efficiency.py --stage inference --workdir /your/new/efficiency_inference
python -B reproduce_efficiency.py --stage latency --workdir /your/new/efficiency_latency
python -B reproduce_efficiency.py --stage training --workdir /your/new/efficiency_training
```

Set `EFFICIENCY_ENV_ROOT` when the matching environments are installed elsewhere
(`deim_cvpr2025/bin/python` and `rtdetr_sota/bin/python`). The profiler is bundled
under `vendor/`. CPU checks are not GPU timing. Measurement scripts and generated
outputs run in the new working copy, leaving the released assets unchanged.
