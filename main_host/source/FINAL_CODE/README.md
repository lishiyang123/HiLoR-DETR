# DEIM and HiLoR main host code

This implementation supports the CDW1630 and BTC experiments using one shared
DEIM/D-FINE-M engine. Dataset-specific profiles retain the reported class counts,
training settings, validation-selected detector checkpoints and checkpoint-specific
CCVR coefficients and value heads. RT-DETR is provided separately.

## Functional organization

The implementation follows two method stages. GCCS provides training-time
complementary supervision; CCVR performs inference-time candidate reranking.
The files below separate functional responsibilities within those stages.

| Stage | Functional file under `engine/deim/` | Responsibility |
| --- | --- | --- |
| GCCS | `auxiliary_geometry.py` | Difficulty-aware auxiliary geometric supervision |
| GCCS | `classification_role.py` | Localization-gain-conditioned classification-role adjustment |
| CCVR | `competition_mapping.py` | Competition-conditioned score mapping |
| CCVR | `candidate_value.py` | Candidate-value estimation, residual score adjustment and remapping |

`ufd`, `lcct`, `qccr` and `umr` are internal identifiers for these functional
subcomponents, not four additional independent method contributions. Existing
configuration keys, tensor keys, class names and import aliases are retained for
compatibility with the published checkpoints and fitting assets. File boundaries
are code organization, not an alternative definition of the method.

## Frozen checkpoint evaluation

Run with the main-host environment recorded in the reproduction release. Accuracy
evaluation uses FP32, EMA weights, batch size 8 and faster-coco-eval 1.7.2.
Use a new output directory outside the frozen release.

```bash
python -u -B reproduce.py --release-root /path/to/HiLoR-DETR \
  --profile cdw_full_seed0_test --output /path/to/new_outputs/cdw_full_seed0

python -u -B reproduce.py --release-root /path/to/HiLoR-DETR \
  --profile btc_full_seed0_test1 --output /path/to/new_outputs/btc_full_seed0_test1
```

Replace `full` with `deim` for the native detector or `gccs` for the detector with
GCCS and native post-processing. Seeds are 0, 1 and 2. BTC has `test1` and `test2`
profiles. Native+CCVR uses `cdw_deim_ccvr_seedN_test` for seeds 0/1/2 and the
Native checkpoint's own fitted mapping and value head. See `../../README.md`.

Each evaluation writes `eval.pth`, `RESULT.json` and the effective configuration.
The metrics include all twelve COCO bbox statistics and AP80/AP85/AP90/AP95.
`--compare-reference` additionally compares the stored evaluation tensors; it
does not change the evaluated model, parameters or output values.

## Detector training

```bash
python -u -B reproduce.py --release-root /path/to/HiLoR-DETR \
  --profile cdw_gccs_train --seed 0 --output /path/to/new_outputs/cdw_gccs_train_seed0
```

Use `cdw_deim_train`, `btc_deim_train` or `btc_gccs_train` for the other training
configurations. These profiles use Train for detector optimization and Val for
checkpoint selection. The common official COCO initialization is retained.
Full HiLoR uses the GCCS-trained detector and subsequently fitted CCVR, not a
different detector-training configuration. New training runs can vary because
CUDA training is not guaranteed to be bitwise deterministic.

Use the profiles above (or the release-root `reproduce_pipeline.py`) as the
executable training entry. Files under `training_records/` record completed runs;
their output-directory paths are not relocation entry points. The launcher uses
FP32 (`use_amp=False`) and writes the effective `config.yml` for each run.

## Frozen candidate fitting

The method tools under `tools/calibration/common/` provide Train/Val capture,
Train-only difficulty-CDF construction, competition mapping freezing and
candidate-value fitting. Mapping target construction and image-level five-fold
fitting are under `tools/qccr/`. Fixed-setting helpers are under
`tools/calibration/fixed_protocol/`. Use `--help` for their explicit input and
output arguments. Detector checkpoints must remain frozen during these stages.

CDW profiles use five categories; BTC profiles use three. Difficulty CDFs are
dataset-specific. Mapping coefficients are fitted on the corresponding Val
outputs, and each value head is fitted on the corresponding frozen Train
outputs. Test is evaluation-only. The recorded fixed settings and fitted assets
are retained; evaluation does not refit or select parameters.

The published training profiles set `ufd_lambda_max=0.75` explicitly; the
criterion's generic constructor default is not the experiment setting. Preserve
the dataset-specific class count and the explicit profile values when rerunning.

## CPU checks

From this source directory, run the synthetic method and mapping-input tests:

```bash
python -B -m unittest discover -s tests -p 'test_method_contracts.py'
python -B -m unittest discover -s tests -p 'test_profile_mapping.py'
```

These tests do not train a detector, load released detector weights, access
dataset images or execute CUDA graphs. They preserve the existing tie rules,
coordinate conventions and numerical implementation.

## Licenses and source identity

Upstream DEIM/D-FINE/RT-DETR notices and `LICENSE` are retained. The final source
file list and hashes are in `SOURCE_MANIFEST.json`. Weights, datasets and fitted
assets remain in their respective reproduction subpackages and are not duplicated
inside the source directory.
