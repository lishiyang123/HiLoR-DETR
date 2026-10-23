# Paper numerical evidence

This package contains source numbers and saved predictions, not publication
graphics or rendering programs. Current photographic examples use BTC seed 0.

| Figure / analysis | Source | Scope |
| --- | --- | --- |
| F1 introductory comparison | `evidence/introduction_comparison/INTRO_FIGURE_EVIDENCE.json` | BTC Test2 `50.jpg`, two paired GT instances |
| F2 detection background | `evidence/framework_example/DETECTION_BACKGROUND.json` | BTC Test1 `492.jpg`; illustrative detections only |
| F7 ranking / crossing / limitation | `evidence/qualitative_cases/QUALITATIVE_CASE_EVIDENCE.json` | BTC Test1 `474.jpg`, Test2 `35.jpg` and `25.jpg` |
| Figure case numbers | `evidence/qualitative_cases/CASE_METADATA.csv` | Unrounded score, IoU and global rank for both methods |
| F6 AP--IoU curves | `evidence/ap_iou_curve/results/curve.csv` | CDW1630 Test, ten thresholds, three-seed method means and paired mean +/- sample SD |
| Oracle geometry | `evidence/oracle/` | DEIM/GSA, seeds 0/1/2; all 300 final-layer query boxes, no class/score/Top-K filter |
| Value-residual sensitivity | `../sensitivity/tau_beta/results/` | CDW1630 Val; 5x5 grid, three value-head fitting seeds |

## AP-IoU curves (CPU)

From the release root, with code + results, NumPy and faster-coco-eval 1.7.2:

```bash
python -B numerical_evidence/evidence/ap_iou_curve/run.py --output /new/external/ap_iou
```

The entry reuses the six shared formal predictions in
`dataset_analysis/inputs.json`; no weights or images are needed. One CSV provides
both panels, including all per-seed values. AP columns are fractions; paired
differences are percentage points. The band is one sample SD, not a confidence
interval. See `evidence/ap_iou_curve/README.md`.

## Oracle geometry (CPU)

From the release root, with code + results and NumPy:

```bash
python -B numerical_evidence/evidence/oracle/run.py --output /new/external/oracle
```

The six compact query-box files and the supplied Test annotation reproduce
per-GT, per-seed and three-seed summaries. No weights, images or GPU are needed.
Each GT uses its best IoU over the complete query pool. This geometric coverage
is not detection recall; definitions and units are in `evidence/oracle/README.md`.

## BTC case protocol

`BTC_CASE_PROTOCOL.json` defines selection and matching. These are post-hoc
illustrative choices from nine inspected BTC test scenes, not exhaustive test-set
ranking and not model, checkpoint or hyperparameter selection. Native photographs
are 1920 x 1200; input locations are in `BTC_INPUT_BINDINGS.json`. The BTC
models were trained and CCVR fitted on BTC, not transferred zero-shot from CDW.

Saved cases are fresh CPU FP32, batch-1 qualitative forwards of the released
seed-0 EMA checkpoints and checkpoint-specific CCVR assets. They are not archived
GPU batch-8 prediction caches, do not replace the formal metric files, and were
exported before display filtering. All 300 final query--class detections are saved.

For F1 and the F7 crossing/limitation panels, show the highest-IoU same-class
candidate with score >= 0.30. For the F7 ranking panel, show the highest-score
same-class valid candidate (IoU >= 0.75, score >= 0.30). Rank is 1-based across
the final global Top-300. The limitation claim is also checked over every
same-class final detection without a score threshold: neither method reaches 0.75.
It is not a claim about all pre-Top-K query boxes. DEIM versus complete HiLoR-DETR
examples do not isolate CCVR's causal effect.

F2 shows actual HiLoR detections with score >= 0.90, without GT-based filtering or
extra NMS. It is a schematic background, not a GSA training-assignment capture.
Do not attach query IDs, Hungarian/residual assignment IoUs or token labels to it.

## Offline numerical replay (standard library only; no GPU)

From the release root, with the results download extracted:

```bash
python -B numerical_evidence/btc_cases.py
```

The command independently recomputes matching, IoU, scores, ranks and the F2
display set from the saved raw predictions and embedded GT, then compares the
current JSON/CSV evidence. It does not draw images or calculate aggregate AP.
Use `--output /new/external/directory` to write the recomputed records separately.

To repeat the CPU qualitative forward, use the main-host environment and the
same released profiles, for example:

```bash
python -B numerical_evidence/export_btc_cases.py --release . --output /new/external/deim_test2 --method deim --split test2 --files 50.jpg 35.jpg 25.jpg
python -B numerical_evidence/export_btc_cases.py --release . --output /new/external/full_test2 --method full --split test2 --files 50.jpg 35.jpg 25.jpg
python -B numerical_evidence/export_btc_cases.py --release . --output /new/external/deim_test1 --method deim --split test1 --files 474.jpg
python -B numerical_evidence/export_btc_cases.py --release . --output /new/external/full_test1 --method full --split test1 --files 474.jpg 492.jpg
```

The BTC case export does not modify model weights, fits or dataset splits.
Source images remain in the BTC data group.
