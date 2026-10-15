# CCVR candidate-value fitting sources

This package reproduces Supplementary Table S5, a Val-only comparison for the
frozen CCVR pipeline on CDW1630. It compares mapping-only CCVR, the primary
Train-capture value estimator, image-level five-fold Val-OOF fitting, and an
equal-image-count Train control for detector seeds 0, 1 and 2.

| Fitting source | AP | AP50 | AP75 | AR100 | Paired AP increment (pp) |
|---|---:|---:|---:|---:|---:|
| Mapping only | 78.122 | 97.213 | 89.698 | 87.723 | — |
| Train captures | 78.328 | 97.269 | 89.943 | 87.723 | +0.207 ± 0.015 |
| Val-OOF | 78.226 | 97.263 | 89.783 | 87.723 | +0.105 ± 0.034 |
| Matched-size Train | 78.233 | 97.261 | 89.796 | 87.723 | +0.111 ± 0.050 |

Metrics are three-detector-seed means; ± is the sample standard deviation of
paired AP increments, not a confidence interval. Held-out detections are pooled
before one COCO evaluation; fold APs are not averaged. Each Val-OOF mapping and
value estimator exclude the target fold. Outer fitting uses 199 or 200 images;
the inner value split uses 179 or 180 fitting images and 20 early-stopping images.
The matched Train control uses the same counts. Settings are fixed at positive
weight 4, head seed 0, threshold 0, residual coefficient 2, and top-k 300.

## CPU verification

From the release root, in the primary-host environment:

```bash
python -B main_host/robustness/ccvr_fitting_sources/scripts/verify.py verify --release .
```

Included assets are the three-seed summary, per-seed fold reports, and thirty
fitted heads. Serialized field names and weight filenames retain compatibility
with the released detector-specific CCVR assets. They do not define additional
proposed modules.

## Train and Val capture inputs

Six large tensor files are not duplicated here. Exact paper inputs are identified
in `bindings/ASSET_BINDINGS.json`. Alternatively, regenerate fresh captures with
the released GCCS checkpoints:

```bash
# Repeat for detector seeds 0, 1 and 2. Only detector capture needs GPU.
python -B reproduce_pipeline.py --host deim --dataset cdw --method full --stage capture --seed 0 \
  --checkpoint main_host/checkpoints/CDW1630/Ours/seed0/best_stg2.pth \
  --workdir /your/new/captures/seed0
```

For arbitrary filenames, `--capture-map` accepts a JSON object keyed by `"0"`,
`"1"`, and `"2"`, each with `"train"` and `"val"` paths. Relative paths resolve
against the map file. `--captures-root` instead discovers seed-specific files.

## CPU preflight and complete fitting-source rerun

Use a new output directory outside the release. With exact original captures:

```bash
python -B main_host/robustness/ccvr_fitting_sources/scripts/run_analysis.py \
  --release . --captures-root /your/captures --output-dir /your/new/fitting_sources --preflight-only

# Remove --preflight-only to fit fold heads and evaluate Val on CPU.
```

No prebuilt input lock is required: the entry generates it from bound release
assets and supplied captures. `build_inputs.py` can also save a standalone lock
with `--output /your/new/input_lock.json`; `--input-lock` remains accepted.
All outputs stay outside the release. Regenerated captures must be explicitly
accepted with `--accept-regenerated-captures`; their actual hashes are saved and
they are not represented as byte-identical historical inputs. The capture loader
still checks split, checkpoint, class count and candidate count. Neither route
retrains the detector, reads Test data, changes published results or searches
parameters.
