# Download preparation and fixed versions

Source repository: <https://github.com/lishiyang123/HiLoR-DETR>.
Versioned code/results/weights and optional data/features:
<https://github.com/lishiyang123/HiLoR-DETR/releases>.

The attached `DOWNLOAD_MANIFEST.json` records the exact contents and checksums
of independently extractable archive parts. Extract the parts needed for the
selected workflow into one directory, preserving relative paths. Numbered
archives are complete tar files; do not concatenate them. Publication splits
large weight/feature groups into smaller attachments. No DOI is claimed.

The authors' original code is Apache-2.0 (`LICENSE`). Third-party code, datasets
and weights retain their own terms; see `THIRD_PARTY_LICENSES.md` and
`DATA_AVAILABILITY.md`. Use these same links in the root README and manuscript.

## Before public upload

- Include `THIRD_PARTY_LICENSES.md`, the root `LICENSE-*.txt` full texts and the
  original license/copyright notices retained in third-party source directories.
  GPL/AGPL-covered components and modifications retain their respective terms.
- The archive groups below describe local reproduction assets, not a declaration
  that every asset has been cleared for public redistribution.
- YOLOv8-L includes its three-seed trained weights, official COCO initializer
  and corresponding Ultralytics source. YOLO11-L likewise includes three trained seeds.
  The provider's CDW-trained YOLOv8-L / YOLO11-L weights are not release inputs.
- Retain the published dataset attribution, license and conversion notices for
  optional dataset downloads. Check model-specific terms before uploading other
  third-party checkpoints. A link to a publicly downloadable file is not itself
  an additional redistribution grant.

## Archive groups

| Group | Contents | Use |
|---|---|---|
| `code` | Source, configurations, installation/data instructions, navigation, manifests | Lightweight executable/documentation download |
| `results` | Predictions, evaluation arrays, numerical tables, logs, analysis outputs and the common Test candidate pool | Offline result checking; no detector training |
| `weights` | Selected detectors, initializers, fitted value heads and comparison heads | Checkpoint inference or training initialization |
| `data` | Bundled dataset images | Optional; see `DATA_AVAILABILITY.md` for originals, licensing and exact reconstruction |
| `features` | Regenerable Train/Val feature captures | Optional; not needed for inference with already fitted CCVR |

This is transport grouping, not deletion or a change to an experiment. Every file
belongs to exactly one group. Archive members retain the original relative paths;
extract the requested groups into **the same directory**. No extra experiment
folder renaming or weight conversion takes place. Required mapping coefficients,
value-head weights, and shared Test candidates retain their existing bindings.

## Downloads by task

Choose **one** source download from the release cited by the manuscript: its Git
tag, GitHub's automatic **Source code (zip/tar.gz)**, or the attached code archive.
Use the matching `DOWNLOAD_MANIFEST.json` to select all other assets. The table
uses these content labels; an asset group may occupy one or several archive parts:

- **code**: the selected release's source and configuration files.
- **results**: its formal predictions, evaluation arrays and numerical results.
- **weights-main-host**: the selected detectors, fitted heads and initialization
  under `main_host/`.
- **weights-btc**: the corresponding assets under `generalization/btc/`.
- **weights-rtdetr-native**: assets under `benchmarks/02_RT_DETR_R18/`.
- **weights-second-host**: assets under `second_host/`.
- **dataset inputs**: either the prepared dataset from the optional data archive,
  or the exact original data prepared using `DATA_AVAILABILITY.md`. Annotations
  are supplied; image-dependent commands still need the corresponding images.

| Task | Required downloads / inputs |
|---|---|
| Saved predictions, numerical tables and BTC figure-number checks | code + results; the matching software dependencies, no weights or image archive |
| Main-host and four-configuration system tables (`main_host/summarize_results.py`) | code + results; main-host CPU environment to read twelve saved evaluation arrays, no weights or images |
| Native + CCVR checkpoint inference | code + results + weights-main-host + CDW images; the Native detector and its checkpoint-specific calibration are under `main_host/checkpoints/CDW1630/B0/seedN/` |
| Main-host raw Test and reusable Train/Val captures | code + results + the optional features parts containing the capture paths in `main_host/manifests/CDW1630_THREE_SEED_RELEASE.json`; no detector forward needed to read them |
| Density/overlap and per-class results (`dataset_analysis/run.py`) | code + results, including the six compressed files under `dataset_analysis/predictions/`; NumPy and faster-coco-eval 1.7.2 on CPU, no weights, image archive or features |
| Three-seed paired Bootstrap (`bootstrap/run.py`) | code + results, including the same six shared predictions under `dataset_analysis/predictions/`; CPU, NumPy and faster-coco-eval 1.7.2, no weights, images or features |
| AP-IoU curves (`numerical_evidence/evidence/ap_iou_curve/run.py`) | code + results, including the same six shared predictions under `dataset_analysis/predictions/`; CPU, NumPy and faster-coco-eval 1.7.2, no weights, images or features |
| Oracle geometry (`numerical_evidence/evidence/oracle/run.py`) | code + results, including six compact full-query box files under `numerical_evidence/evidence/oracle/results/boxes/`; CPU and NumPy only, no weights, images, features or COCO evaluator |
| CCVR fitting-source tables / Val metric replay (`main_host/robustness/ccvr_fitting_sources/run.py`) | code + results; CPU; saved per-seed metrics or twelve pooled prediction files; no weights, images or captures |
| CCVR fitting-source fold-head fitting (`--stage fit`) | code + results + shared GCCS Train/Val captures in features + the three formal Train-fitted value heads in weights; CPU; no detector weights, images or Test inputs |
| CCVR coefficients, stage increments and matching/ranking (`main_host/analysis/ccvr/run.py`) | code + results + four shared seed-0 raw Test captures in features; main-host CPU environment; no detector weights, images or detector forward |
| Seven-row candidate-ranking table from saved predictions (`--stage metrics`) | code + results; CPU, no model weights or images |
| Five external ranking methods from the common Test pool (`--stage replay`) | code + results + the weights part containing `ranking_comparison/state/fitted/iou_regression/iou_regressor_seed123.pt`; main-host CUDA environment, no detector forward |
| Repeat ranking-method fitting and Val selection (`--stage all`) | the replay inputs + the features part containing `sensitivity/tau_beta/inputs/train_capture.pt` and `val_capture.pt`; these captures are shared, not duplicated in the comparison package |
| Main DEIM/GCCS/complete-method checkpoint inference | code + results + weights-main-host + CDW dataset inputs |
| BTC checkpoint inference | code + results + weights-btc + BTC dataset inputs |
| Native RT checkpoint inference | code + weights-rtdetr-native + CDW dataset inputs |
| RT GCCS checkpoint inference | code + results + weights-second-host + CDW dataset inputs |
| Complete RT checkpoint inference | code + results + weights-second-host + CDW dataset inputs |
| RT `verify_inputs.py` (all seeds/modes) | code + results + weights-rtdetr-native + weights-second-host; checks saved evaluations/configuration/weights, no image archive or GPU |
| RT CPU suite: `cd second_host` then `python -B -m unittest discover -s tests -p 'test_*.py'` | code + results + the weight part containing `main_host/pretrained/deim_hgnetv2_m_coco.pth`; synthetic test fixtures are created temporarily, no GPU or feature archive |
| Fresh main-host training, capture, fitting and evaluation | code + results + weights-main-host (including official initialization) + the selected dataset inputs; stages create new captures and fits outside the release |
| Fresh RT training, capture, fitting and evaluation | code + results + weights-rtdetr-native (includes official COCO initialization) + CDW dataset inputs; stages create new captures and fits outside the release |
| DEIM/GCCS/full HiLoR paired FP32 latency | code + results + weights-main-host + CDW dataset inputs; main-host environment and RTX 4090, all three configurations in one session |

The complete RT inference binding is
`second_host/artifacts/inference/seedN/qccr_umr_binding.json`. This JSON and
its fitting metadata are in **results**, not the RT weight archive. The
checkpoint and fitted value-head tensors are in **weights-second-host**.
Download both groups for `--mode full`; source plus weights alone is not a
complete input set. GCCS evaluation also constructs the loss object and reads
its fixed Train difficulty CDF from the results archive. Train/Val **features** are optional for inference with
supplied fits and for a fresh capture-and-fit run that regenerates them.

The full source-workflow regression suite at
`main_host/source/FINAL_CODE/tests/` additionally requires the main-host
initializer, the RT COCO initializer and `second_host/runs/seed0/best_val.pth`;
obtain the weights-main-host, weights-rtdetr-native and weights-second-host
groups before running it. These asset-dependent tests are distinct from
saved-prediction numerical replay.

Each weight part is independently extractable. Obtain every part containing the
required assets, using the selected release's download inventory and links. Extract
the chosen source and all required asset parts into **one** release root, not
nested version directories. No part is concatenated with another part.

Within each archive, byte-identical files use standard tar hard links to an
earlier member. Every input is still checked against its own SHA-256 before a
link is written. Extraction recreates every original path and its exact bytes;
there is no cross-archive link dependency. This avoids duplicate payloads without
changing runtime paths, configurations or scientific assets. Catalog sizes report
both logical file bytes and unique payload bytes.

## Prepare a download inventory (no compression)

From the aggregate release root, using Python 3.10+:

```bash
python -B prepare_downloads.py
python -B prepare_downloads.py --output /your/new/download_plan
```

The first command prints the release ID and exact uncompressed group sizes. The
second additionally writes `DOWNLOAD_MANIFEST.json` outside the release. It does
not create archives, train, fit, infer, or alter the frozen package. The catalog
contains the group membership and original file digests; archive checksums are
added only after an actual build.

## Build archives when a publication destination is ready

```bash
python -B prepare_downloads.py --build --groups code results weights --output /your/new/submission_downloads
```

Add `data features` to `--groups` only if those optional downloads are wanted.
Compression consumes CPU and additional disk space; this is not a quick GPU
check. The destination must be new and outside the release. All input file hashes
are checked while archiving, archives contain no absolute/parent paths, and the
release is never modified. The download manifest records the actual compressed
size and SHA-256 of each built archive. Files ending in `.part` indicate an
incomplete build and must not be published.

Upload only assets approved by the public-upload checklist above, together with
an inventory of the assets actually supplied. A local complete archive is not
automatically a public-upload candidate. If an asset is obtained from its
original provider instead, keep the exact filename, source and expected digest
in the download instructions. Review the inventory before upload; provide a
smaller or split optional download if the host imposes size limits.

## Assemble and verify

This is a Linux/CUDA reproduction package. Extract into a case-sensitive Linux
filesystem (or a case-sensitive WSL workspace): upstream and result directories
may contain case-distinct filenames. Do not unpack and repack the full release
through a normal case-insensitive Windows directory, which can merge those names.

Extract groups into a single clean directory, preserving their member paths.
Recreate missing dataset images using the pinned original data and the released
data-preparation instructions. Include optional feature captures if a full
release-wide hash check, rather than checkpoint inference, is required. Then:

```bash
python -B verify_release.py verify
```

The full verifier checks all included files and deliberately reports missing files
in a partial download. This is not an inference failure: install the required
groups/data for the selected task using `REPRODUCTION_QUICKSTART.md`. Retraining
remains a statistical reproduction; archived-result verification and checkpoint
inference are distinct workflows.

## Comparison inputs

Saved predictions are in the results group; detector inference requires the
corresponding selected weights. YOLOv8-L and YOLO11-L use the bundled trained
checkpoints, not the provider's CDW-trained checkpoints. Their official COCO
initializers are included for fresh training.

