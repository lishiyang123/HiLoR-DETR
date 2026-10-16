# Paper numerical evidence

This package contains source numbers and saved predictions, not publication
graphics or rendering programs. Current photographic examples use BTC seed 0.

| Figure / analysis | Source | Scope |
| --- | --- | --- |
| F1 introductory comparison | `evidence/introduction_comparison/INTRO_FIGURE_EVIDENCE.json` | BTC Test2 `50.jpg`, two paired GT instances |
| F2 detection background | `evidence/framework_example/DETECTION_BACKGROUND.json` | BTC Test1 `492.jpg`; illustrative detections only |
| F7 ranking / crossing / limitation | `evidence/qualitative_cases/QUALITATIVE_CASE_EVIDENCE.json` | BTC Test1 `474.jpg`, Test2 `35.jpg` and `25.jpg` |
| Figure case numbers | `evidence/qualitative_cases/CASE_METADATA.csv` | Unrounded score, IoU and global rank for both methods |
| AP--IoU curve and oracle geometry | `evidence/ap_iou_curve/` | CDW1630 quantitative analyses |
| Parameter sensitivity | `evidence/parameter_sensitivity/` | CDW1630 Val / OOF-Val; distinct evaluation scopes |

## BTC case protocol

`BTC_CASE_PROTOCOL.json` defines selection and matching. These are post-hoc
illustrative choices from nine inspected BTC test scenes, not exhaustive test-set
ranking and not model, checkpoint or hyperparameter selection. Native photographs
are 1920 x 1200; locations and hashes are in `BTC_INPUT_BINDINGS.json`. The BTC
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
extra NMS. It is a schematic background, not a GCCS training-assignment capture.
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

Model weights, fits, dataset splits and quantitative result tables are unchanged.
`SOURCE_BINDINGS.json` records source provenance; current file identity is checked
by the package and release manifests. Source images remain in the BTC data group.
The separate `qualitative_evidence/` package contains the CDW1630 descriptive
prediction/ranking study; its reporting-case IDs are not the current BTC figures.
