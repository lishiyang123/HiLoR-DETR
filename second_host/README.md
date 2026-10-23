# RT-DETR-R18 second-host reproduction

The release contains Native, GCA and complete HiLoR-DETR for detector seeds
0/1/2. GCA and the complete method share a detector checkpoint; the complete
method adds the CCVR coefficients and value head fitted to that checkpoint.

## Install and download

Follow [INSTALL.md](INSTALL.md) to create the RT environment, including
`faster-coco-eval==1.7.2`. Use the
[task-to-download table](../DISTRIBUTION.md#downloads-by-task) to assemble inputs.
Full RT inference needs **code + results + weights-rt + dataset inputs**;
the results archive contains the required binding JSON and fitting metadata,
while weights-rt contains the selected detector and value-head weights.
Native inference uses weights-rt instead. Train/Val feature captures
are not needed to evaluate already fitted models. The fixed Train difficulty CDF
loaded during criterion construction is included with the code in
`inputs/difficulty_cdf_train640.json`.

## Method source organization

GCA and CCVR are the two method stages. Under
`source/RT-DETR_HiLoR/rtdetr_pytorch/src/zoo/rtdetr/`, their implementation is:

| Stage / responsibility | Source file |
| --- | --- |
| GCA orchestration and classification-role adjustment | `hilor_gca.py` |
| GCA auxiliary geometric supervision | `auxiliary_geometry.py` |
| CCVR competition-conditioned mapping | `competition_mapping.py` |
| CCVR candidate-value estimation | `candidate_value.py` |
| Full inference post-processing | `hilor_postprocessor.py` |

`ufd`, `qccr` and `umr` are implementation/serialization identifiers for these
functional responsibilities, not three additional independent method claims.
The physical files use the same functional names as the main host. Serialized
keys and import aliases remain compatible with the fitted artifacts. File
organization does not change the two-stage method definition.

Inference validates the detector-specific mapping, value-head weights, fitting
identities and model dimensions. It does not require a source-code or development
protocol lock. The optional release-wide integrity check covers source files;
fitting records retain their experimental input identities.

## Results and assets

- `paper_tables/RTDETR_PER_SEED.csv`: individual seeds and AP80/AP85/AP90/AP95.
- `paper_tables/RTDETR_MEAN_STD_AND_DELTA.csv`: three-seed means, sample standard
  deviations and stage increments for all 12 COCO and four strict-IoU metrics.
- `evaluations_gca/` and `evaluations/`: saved Test predictions and COCO arrays.
- `runs/seedN/`: selected GCA weights, completed training logs and Val selection.
- `artifacts/inference/seedN/`: separately fitted, checkpoint-specific CCVR assets.
- `artifacts/captures/seedN/`: complete Train/Val candidate captures.
- `../benchmarks/02_RT_DETR_R18/`: Native source, weights and
  Test results, shared with the SOTA comparison rather than duplicated here.
- `CURRENT_RESULT_MANIFEST.json`: the checkpoint-to-result map.

Complexity, latency and training cost are maintained in `../efficiency/`.

Training uses the official COCO initialization, 102 epochs, batch 4, FP32 and
EMA. Val/Test batch is 8. Checkpoints maximize Val AP, retaining the earlier
epoch on exact ties. Native and GCA use the same initialization, optimization,
augmentation and evaluation settings. The fixed host adaptation is
alpha_u_max=0, lambda_ufd_max=0.75, e0=20 and ramp=5.

Training Val, capture statistics and formal Test use faster-coco-eval 1.7.2 in
the RT environment (Torch 2.0.1+cu118). Test inference is deterministic with
FP32/EMA, CPU threads=1, cuDNN benchmarking disabled, deterministic cuDNN and
PyTorch algorithms, and CUBLAS_WORKSPACE_CONFIG=:4096:8. This does not promise
bitwise identity across different hardware/library versions or fresh retraining.

## Reproduce stored results

Run from this directory in the RT environment. Use fresh outputs outside the release.

```bash
python -B reproduce_second_host.py --mode full --seed 0 --preflight-only
python -B reproduce_second_host.py --mode full --seed 0 --output /your/new/rt_full_seed0
python -B summarize_test_results.py --output /your/new/rt_tables
```

Modes are `native`, `gca` and `full`. Repeat for detector seeds 1 and 2.
GPU inference loads the selected EMA detector and checkpoint-bound fits. Table
extraction and saved-prediction replay require no GPU. From the release root:

```bash
python -B evaluate_test_predictions.py --predictions second_host/evaluations/seed0/predictions.json --annotation main_host/datasets/CDW1630/annotations/instances_test2017.json --output /your/new/rt_replay
```

## Fresh training, capture and fitting

Use `train.py` for Native/GCA training or the release-root staged entry:

```bash
python -B train.py --method native --seed 0 --workdir /your/new/native_seed0
python -B train.py --method gca --seed 0 --workdir /your/new/gca_seed0
```

From the release root, capture and fit the GCA checkpoint:

```bash
python -B reproduce_pipeline.py --host rtdetr --dataset cdw --method full --seed 0 --stage capture --workdir /your/new/gca_seed0 --checkpoint /your/new/gca_seed0/train/best_val.pth
python -B reproduce_pipeline.py --host rtdetr --dataset cdw --method full --seed 0 --stage fit --workdir /your/new/gca_seed0 --checkpoint /your/new/gca_seed0/train/best_val.pth
python -B reproduce_pipeline.py --host rtdetr --dataset cdw --method full --seed 0 --stage evaluate --workdir /your/new/gca_seed0 --checkpoint /your/new/gca_seed0/train/best_val.pth
```

Each stage runs explicitly and writes outside the release. CCVR settings are
lambda=0.001, positive weight=4, head seed=0, tau=0, beta=2, Top-K=300. Mapping
coefficients use Val; value-head fitting and early stopping use 938 and 104
images within Train. Test never selects these settings. See `docs/` for scopes.

For current input validation and CPU regression tests:

Run these commands from `second_host/`. The complete test suite also reads
`main_host/pretrained/deim_hgnetv2_m_coco.pth`, supplied by the weight part
containing that path in the release download inventory, for its cross-host workflow regression. Code and
results alone are sufficient for saved-prediction replay, not this asset-dependent
test. `verify_inputs.py` checks selected inputs for all three RT modes and needs
both RT weight groups and results as listed in the download table; it does not
load dataset images.

```bash
python -B verify_inputs.py
python -B -m unittest discover -s tests -p 'test_*.py'
```

The public workflow uses `tools/training.py` and `tools/capture.py` for shared
training and Train/Val capture routines. Outputs always belong to the supplied
external work directory. Capture/fitting receipts preserve their original input
identities; the release-wide manifest describes the files shipped in this version.
