# Download preparation and fixed versions

Source repository: <https://github.com/lishiyang123/HiLoR-DETR>.
Versioned code/results/weights and optional data/features:
<https://github.com/lishiyang123/HiLoR-DETR/releases>.

The attached `DOWNLOAD_MANIFEST.json` records the contents of independently extractable archive parts. Extract the parts needed for the
selected workflow into one directory, preserving relative paths. Numbered
archives are complete tar files; do not concatenate them. Publication splits
large weight/feature groups into smaller attachments. No DOI is claimed.

The authors' original code is Apache-2.0 (`LICENSE`). Third-party code, datasets
and weights retain their own terms; see `THIRD_PARTY_LICENSES.md` and
`DATA_AVAILABILITY.md`. Use these same links in the root README and manuscript.

## Archive groups

| Group | Contents | Use |
|---|---|---|
| `code` | Source, configurations, installation/data instructions, navigation, manifests | Lightweight executable/documentation download |
| `results` | Predictions, evaluation arrays, numerical tables, logs, analysis outputs and the common Test candidate pool | Offline result checking; no detector training |
| `weights` | Selected detectors, main-host initializer, fitted value heads and comparison heads | Checkpoint inference or training initialization |
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
- **weights-main**: the selected detectors, fitted heads and initialization
  under `main_host/`.
- **weights-btc**: the corresponding assets under `generalization/btc/`.
- **weights-rt**: assets under `benchmarks/02_RT_DETR_R18/`.
- **weights-rt**: assets under `second_host/`.
- **dataset inputs**: either the prepared dataset from the optional data archive,
  or the exact original data prepared using `DATA_AVAILABILITY.md`. Annotations
  are supplied; image-dependent commands still need the corresponding images.

| Task | Required downloads / inputs |
|---|---|
| Saved predictions, numerical tables and BTC figure-number checks | code + results; the matching software dependencies, no weights or image archive |
| Main-host and four-configuration system tables (`main_host/summarize_results.py`) | code + results; main-host CPU environment to read twelve saved evaluation arrays, no weights or images |
| Native + CCVR checkpoint inference | code + results + weights-main + CDW images; the Native detector and its checkpoint-specific calibration are under `main_host/checkpoints/CDW1630/B0/seedN/` |
| Main-host raw Test and reusable Train/Val captures | code + results + the optional features parts containing the capture paths in `main_host/manifests/CDW1630_THREE_SEED_RELEASE.json`; no detector forward needed to read them |
| Density/overlap and per-class results (`dataset_analysis/run.py`) | code + results, including the six compressed files under `dataset_analysis/predictions/`; NumPy and faster-coco-eval 1.7.2 on CPU, no weights, image archive or features |
| Three-seed paired Bootstrap (`bootstrap/run.py`) | code + results, including the same six shared predictions under `dataset_analysis/predictions/`; CPU, NumPy and faster-coco-eval 1.7.2, no weights, images or features |
| AP-IoU curves (`numerical_evidence/evidence/ap_iou_curve/run.py`) | code + results, including the same six shared predictions under `dataset_analysis/predictions/`; CPU, NumPy and faster-coco-eval 1.7.2, no weights, images or features |
| Oracle geometry (`numerical_evidence/evidence/oracle/run.py`) | code + results, including six compact full-query box files under `numerical_evidence/evidence/oracle/results/boxes/`; CPU and NumPy only, no weights, images, features or COCO evaluator |
| CCVR fitting-source tables / Val metric replay (`main_host/robustness/ccvr_fitting_sources/run.py`) | code + results; CPU; saved per-seed metrics or twelve pooled prediction files; no weights, images or captures |
| CCVR fitting-source fold-head fitting (`--stage fit`) | code + results + shared GSA Train/Val captures in features + the three formal Train-fitted value heads in weights; CPU; no detector weights, images or Test inputs |
| CCVR coefficients, stage increments and matching/ranking (`main_host/analysis/ccvr/run.py`) | code + results + four shared seed-0 raw Test captures in features; main-host CPU environment; no detector weights, images or detector forward |
| Seven-row candidate-ranking table from saved predictions (`--stage metrics`) | code + results; CPU, no model weights or images |
| Five external ranking methods from the common Test pool (`--stage replay`) | code + results + the weights part containing `ranking_comparison/state/fitted/iou_regression/iou_regressor_seed123.pt`; main-host CUDA environment, no detector forward |
| Repeat ranking-method fitting and Val selection (`--stage all`) | the replay inputs + the features part containing `sensitivity/tau_beta/inputs/train_capture.pt` and `val_capture.pt`; these captures are shared, not duplicated in the comparison package |
| Main DEIM/GSA/complete-method checkpoint inference | code + results + weights-main + CDW dataset inputs |
| BTC checkpoint inference | code + results + weights-btc + BTC dataset inputs |
| Native RT checkpoint inference | code + weights-rt + CDW dataset inputs |
| RT GSA checkpoint inference | code + results + weights-rt + CDW dataset inputs |
| Complete RT checkpoint inference | code + results + weights-rt + CDW dataset inputs |
| RT `verify_inputs.py` (all seeds/modes) | code + results + weights-rt + weights-rt; checks saved evaluations/configuration/weights, no image archive or GPU |
| RT CPU suite: `cd second_host` then `python -B -m unittest discover -s tests -p 'test_*.py'` | code + results + the weight part containing `main_host/pretrained/deim_hgnetv2_m_coco.pth`; synthetic test fixtures are created temporarily, no GPU or feature archive |
| Fresh main-host training, capture, fitting and evaluation | code + results + weights-main (including official initialization) + the selected dataset inputs; stages create new captures and fits outside the release |
| Fresh RT training, capture, fitting and evaluation | code + results + the separately downloaded official RT COCO initializer + CDW dataset inputs; stages create new captures and fits outside the release |
| DEIM/GSA/full HiLoR paired FP32 latency | code + results + weights-main + CDW dataset inputs; main-host environment and RTX 4090, all three configurations in one session |

The complete RT inference binding is
`second_host/artifacts/inference/seedN/qccr_umr_binding.json`. This JSON and
its fitting metadata are in **results**, not the RT weight archive. The
checkpoint and fitted value-head tensors are in **weights-rt**.
Download both groups for `--mode full`; source plus weights alone is not a
complete input set. GSA evaluation also constructs the loss object and reads
its fixed Train difficulty CDF included with the code. Train/Val **features** are optional for inference with
supplied fits and for a fresh capture-and-fit run that regenerates them.

The full source-workflow regression suite at
`main_host/source/FINAL_CODE/tests/` additionally requires the main-host
initializer, the RT COCO initializer and `second_host/runs/seed0/best_val.pth`;
obtain the weights-main and weights-rt groups plus the
official RT initializer linked in `benchmarks/CONFIGURATION_GUIDE.md`. These asset-dependent tests are distinct from
saved-prediction numerical replay.

Each weight part is independently extractable. Obtain every part containing the
required assets, using the selected release's download inventory and links. Extract
the chosen source and all required asset parts into **one** release root, not
nested version directories. No part is concatenated with another part.

## Extract the archives

Use a case-sensitive Linux or WSL filesystem. Extract the selected archives into
one directory, preserving their relative paths. Standard tar hard links may be
used within an archive; no link depends on another archive.

Use `REPRODUCTION_QUICKSTART.md` for the execution commands. Saved-result replay,
checkpoint inference and fresh training require different inputs, as listed above.

## Comparison inputs

Saved predictions are in the results group; detector inference requires the
corresponding selected weights. YOLOv8-L and YOLO11-L use the bundled trained
checkpoints, not the provider's CDW-trained checkpoints. Their official COCO
initializers are linked in `benchmarks/CONFIGURATION_GUIDE.md` for fresh training.

