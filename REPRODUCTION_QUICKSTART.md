# Reproduction quick start

## Install the matching environment and inputs

- Main DEIM/BTC host: [installation commands](main_host/environment/CDW1630_B0_REFERENCE/INSTALL.md).
- RT-DETR Native/GCA/complete method: [installation commands](second_host/INSTALL.md),
  including the required `faster-coco-eval==1.7.2` dependency.
- Comparison models: use their model-specific environment records and
  [configuration guide](benchmarks/CONFIGURATION_GUIDE.md).

Before running a command, use the [task-to-download table](DISTRIBUTION.md#downloads-by-task).
The source tag, automatic Source code ZIP/TAR and attached code archive are
equivalent source downloads; none includes model weights or dataset images.
RT complete-model inference additionally needs binding JSON files from the
results archive. Downloaded parts must be extracted into the same release root.

## Data access and raw-data preparation

See `DATA_AVAILABILITY.md` for version-specific CDW1630/BTC downloads,
citations, licenses and fixed split responsibilities. The supplied COCO
annotations are ready for direct reproduction. To rebuild them independently
from original YOLO/VOC labels, use
`main_host/data_preparation/prepare_dataset.py`; its README
provides both commands. Reconstruction needs Python 3.10+ and Pillow, no
GPU, and verifies byte-identical annotation output without replacing assets.

## Choose the evidence scope

- Main CDW1630 results: `main_host/README.md`
- Four-configuration system comparison: `main_host/README.md`
- BTC generalization: `generalization/btc/README.md`
- External-model comparison: `benchmarks/README.md`
- RT-DETR-R18 Native/GCA/full method: `second_host/README.md`
- Complexity, paired FP32 latency and training cost: `efficiency/README.md`
- Sensitivity, bootstrap, qualitative and dataset analyses: use the README in the
  corresponding top-level package.

## Path portability

The release verifier and reviewer navigation are relocation-safe. Archived run records
may retain absolute paths from the machine on which they were produced; these paths document
the original environment and do not need to be recreated. Use each package's documented
root and environment overrides when running a reproduction entry point. Package-specific
software environments are listed in the corresponding environment records.

## Evaluation boundary

Test data were not used to choose checkpoints, training parameters, inference parameters,
or formal evaluation thresholds. Held-out test outputs generated after settings were fixed,
paired bootstrap, and the disclosed qualitative analysis are reporting stages rather than
model-selection stages.

Current formal Test replay uses **faster-coco-eval 1.7.2**. RT Native/GCA training Val, capture statistics and Test all use this backend.
Other host environments are documented in ENVIRONMENT_MATRIX.md.
RT checkpoint Test inference uses Torch 2.0.1+cu118, FP32, EMA, batch 8 and
deterministic execution: cuDNN benchmarking disabled, deterministic cuDNN and
PyTorch algorithms enabled, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. The entry sets
these controls before GPU execution, with CPU/OpenMP threads pinned to 1.
Published checkpoint identities, separately fitted CCVR assets, Test predictions
and metrics are listed in second_host/CURRENT_RESULT_MANIFEST.json.

CPU-only replay of the stored Native RT-DETR or D-FINE predictions:

```bash
python -B evaluate_test_predictions.py --model rtdetr-r18 --seed 0 --output /your/new/rt_replay
python -B evaluate_test_predictions.py --model dfine-m --seed 0 --output /your/new/dfine_replay
```

Use `--predictions ... --annotation ...` for another saved COCO prediction file.
The RT checkpoint entry defaults to the same accelerated backend and also saves
`predictions.json`; `--evaluator pycocotools` requests the historical backend.
No global pycocotools monkey-patch is applied. Saved-prediction replay requires no
GPU. For RT tables use `second_host/summarize_test_results.py`;
it reads the nine published evaluation arrays and regenerates the three-stage,
twelve-COCO and strict-IoU statistics. Independent repeated runs of all nine
checkpoint/mode cases agree exactly in the documented environment. This does
not promise bitwise GPU agreement across different hardware or library versions.

## Portable executable defaults

Reviewer-facing executable entry points derive the release root from their own file
location and default to the Python interpreter that launched them. A reviewer may use
the following explicit overrides when separate environments are required:

- `FINAL_PAPER_RELEASE_ROOT`: relocated release root;
- `BTC_REPRO_PYTHON`: BTC generalization interpreter;
- `PYTHON`: generic launcher fallback.

Files under `materialized_configs/`, `configs/materialized_current_server/`,
`configs/final/`, and formal run/evidence directories preserve the paths recorded by
the original experiment. They preserve the original experiment settings and are not used to discover the
relocated release root.

## Study inputs and offline replay

```bash
python -B reviewer_studies.py list
python -B reviewer_studies.py check-inputs
python -B second_host/reproduce_second_host.py --mode full --seed 0 --preflight-only
```

Use the main-host environment for the seed-0 candidate-ranking comparison:

```bash
python -B ranking_comparison/run.py --stage metrics --output /your/new/ranking_metrics
python -B reviewer_studies.py nearest-replay --output /your/new/ranking_replay
```

The first command recomputes all seven rows from compressed saved predictions
on CPU. The second replays the five external methods on the common Test capture,
using CUDA FP32 postprocessing and the supplied fits; it neither trains nor
forwards a detector. Native GCA and complete CCVR use the saved main-pipeline
reference outputs. See `ranking_comparison/README.md` for Val selection, fitting
commands and the optional shared Train/Val captures.

For GPU checkpoint evaluation use the second-host release wrapper; for GPU
complexity/latency/training-cost reruns use the explicit efficiency stages.
Do not run every GPU stage automatically. Use the staged workflow below for
fresh training and checkpoint-specific fitting; outputs stay outside the release.

## Paired FP32 inference latency

From the release root in the main-host environment, with the required inputs
from `DISTRIBUTION.md`:

```bash
EFFICIENCY_TMP_ROOT=/tmp/hilor_eff python -B efficiency/reproduce_efficiency.py \
  --stage latency --env-root /path/to/envs --workdir /your/new/paired_latency
```

`/path/to/envs` contains `deim_cvpr2025/bin/python`; use the actual installation
location. This command measures DEIM, GCA and full HiLoR together on an RTX 4090:
batch 1, 640x640, 237 real Test images, 50 warmups per configuration and three
balanced alternating passes. Detector forward and complete postprocessing use
FP32 tensors with autocast disabled. Matrix-multiplication TF32 is disabled;
cuDNN TF32 is enabled, matching the main-host AP environment. Timing excludes
IO, decoding, preprocessing, H2D transfer and COCO evaluation. It does not compute AP.

The authoritative table is `efficiency/publication/LATENCY_TABLE.csv`;
its raw measurements are under `efficiency/outputs/latency/`.
Complexity and training cost are separate stages and tables; see
[`efficiency/README.md`](efficiency/README.md) for their commands and scopes.

## Shared CDW1630 / BTC source

`main_host/source/FINAL_CODE/` is the main-host implementation.
Use its `reproduce.py` profiles for checkpoint evaluation or fresh detector
training; its README lists the supported profiles.
RT-DETR uses its independent second-host entry. The release-wide manifest binds
all current source files and numerical assets.

## Fresh training → fitting → evaluation

Use `reproduce_pipeline.py` from the release root, in the matching main-host or
RT-DETR environment. Each command runs only the specified stage. Choose a new
work directory outside the release, separately for every host/dataset/seed.

```bash
python -B reproduce_pipeline.py --host rtdetr --dataset cdw --method full --seed 0 --stage train --workdir /your/new/rtdetr_seed0 --preflight-only
python -B reproduce_pipeline.py --host rtdetr --dataset cdw --method full --seed 0 --stage train --workdir /your/new/rtdetr_seed0
python -B reproduce_pipeline.py --host rtdetr --dataset cdw --method full --seed 0 --stage capture --workdir /your/new/rtdetr_seed0 --checkpoint /your/new/rtdetr_seed0/train/best_val.pth
python -B reproduce_pipeline.py --host rtdetr --dataset cdw --method full --seed 0 --stage fit --workdir /your/new/rtdetr_seed0 --checkpoint /your/new/rtdetr_seed0/train/best_val.pth
python -B reproduce_pipeline.py --host rtdetr --dataset cdw --method full --seed 0 --stage evaluate --workdir /your/new/rtdetr_seed0 --checkpoint /your/new/rtdetr_seed0/train/best_val.pth
```

For the main host use `--host deim --dataset cdw` or `--host deim --dataset btc`.
Main-host training preserves the original stage-1/stage-2 checkpoint convention;
explicitly supply the checkpoint selected by **Val AP**, not merely the last
checkpoint. BTC evaluation requires `--split test1` or `--split test2`.

`--method native` and `--method gca` require only train/evaluate. The full method
trains the GCA detector, captures only Train/Val, fits its **own** CCVR assets,
then evaluates. The published fixed tuple remains lambda=0.001, head seed=0,
positive weight=4, tau=0, beta=2, Top-K=300. Head early stopping uses an internal
Train holdout, not Val or Test. Five-fold mapping diagnostics do not select a
new lambda. `--preflight-only` never trains, fits, infers or writes outputs.

Direct new-checkpoint evaluation is also available through main-host
`reproduce.py --checkpoint ... --ccvr-binding ...` and RT
`reproduce_second_host.py --checkpoint ... --binding ...`. Rescoring requires
new checkpoint-bound fits; without overrides both entries replay the stored
paper assets. Retraining is a statistical reproduction, not a guarantee of
bitwise-identical numbers across hardware/environments.

### Native DEIM + CCVR control

The CDW control fits CCVR to the Native detector's own Train/Val outputs.
Use `--host deim --dataset cdw --method native-ccvr` with the same four stages
above, in its own external work directory. Its `train` stage uses the Native
training profile; `capture` exposes that checkpoint's final hidden states;
`fit` uses only its Train/Val captures; `evaluate` selects the
`cdw_deim_ccvr_seedN_test` profile with the new `FIT_BINDING.json`.
Do not reuse GCA captures, mapping coefficients or a GCA value head.
The control is not declared for BTC or the RT host.

## Main-host tables and reusable inputs

Recompute all main-host seed and stage tables on CPU:

```bash
python -B main_host/summarize_results.py --output /new/external/main_tables
```

This reads the twelve saved formal evaluation arrays, not detector inference.
Use the main-host environment. Input paths, including shared predictions and
optional raw captures, are listed in
`main_host/manifests/CDW1630_THREE_SEED_RELEASE.json`. The same command also
regenerates `system_per_seed.csv`, `system_summary.csv` and `system_deltas.csv`
for DEIM, DEIM+CCVR, GCA and complete HiLoR-DETR. Native+CCVR inference uses
`cdw_deim_ccvr_seedN_test` for N=0/1/2 and the Native detector's own fitted assets.

## CCVR fitting-source analysis on Val

```bash
python -B main_host/robustness/ccvr_fitting_sources/run.py --stage tables --output /new/external/fitting_tables
python -B main_host/robustness/ccvr_fitting_sources/run.py --stage metrics --output /new/external/fitting_metrics
```

For fold-head fitting, use `--stage fit` with the shared Train/Val captures and
the three formal Train-fitted heads. All stages use CPU; Test is not an input.
See `main_host/robustness/ccvr_fitting_sources/README.md` for inputs and settings.

## CCVR gain diagnostics (seed 0)

```bash
python -B main_host/analysis/ccvr/run.py --release . --output /new/external/ccvr_seed0
```

The CPU entry uses shared formal predictions and raw Test captures. It reports
mapping coefficients, stage increments and COCO one-to-one matching/ranking,
without detector inference or refitting. See `main_host/analysis/ccvr/README.md`.

## Value-residual sensitivity

See `sensitivity/README.md`. Replay the saved heads on Val (CPU):

```bash
python -B sensitivity/tau_beta/run.py --output /new/external/tau_beta
```

The 5x5 grid uses delta=0.02 and three value-head fitting seeds. Add `--refit-heads` to repeat fitting from the saved Train capture.

## Paired image Bootstrap

Recompute DEIM versus HiLoR-DETR for seeds 0, 1 and 2 separately on CPU:

```bash
python -B bootstrap/run.py --output /new/external/bootstrap
```

Each seed uses 5,000 paired image resamples and a two-sided 95% percentile
interval. Required inputs are code + results, including the six shared
predictions listed in `dataset_analysis/inputs.json`; no weights or images are
needed. See `bootstrap/README.md` for dependencies and result columns.

## AP-IoU curves and paired differences

Recompute the two Fig. 6 panels on CPU from the same six shared predictions:

```bash
python -B numerical_evidence/evidence/ap_iou_curve/run.py --output /new/external/ap_iou
```

Requires code + results, NumPy and faster-coco-eval 1.7.2; no weights or images.
`curve.csv` contains the ten-threshold curves for seeds 0/1/2, method means,
and paired HiLoR-minus-DEIM means and sample SDs. The shaded band is +/- one
sample SD, not a Bootstrap confidence interval. See the adjacent README for
column units.

## Oracle geometric analysis

Recompute DEIM versus GCA for all three detector seeds on CPU:

```bash
python -B numerical_evidence/evidence/oracle/run.py --output /new/external/oracle
```

Requires code + results and NumPy. The six compact full-query box files are
included; no weights, images, feature captures, PyTorch or GPU are needed.
The output contains per-GT, per-seed and three-seed summaries. Oracle coverage
is not detection recall; see the adjacent README for definitions and units.

## Dataset statistics

Recompute split, category, scale and density tables, plus Test groups using a fixed density cutoff of 5 and a Train-derived overlap cutoff:

```bash
python -B dataset_analysis/source/analyze_dataset.py --project . --output /new/external/data_stats
```

This command uses annotations and image paths only, requires no GPU,
and writes full-precision numbers to a new external directory.

Recompute the four-group and per-class comparisons from the six shared formal
prediction files (CPU, NumPy and faster-coco-eval 1.7.2):

```bash
python -B dataset_analysis/run.py --stage all --output /new/external/test_analysis
```

Use `--stage groups` or `--stage classes` for one analysis only. Required inputs
are code + results; no image archive, model weights or feature captures are used.
See `dataset_analysis/README.md` for the fixed density rules, overlap definition
and result locations.
