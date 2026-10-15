# Benchmark assets

This directory contains the comparison-detector implementations, configurations,
selected weights, predictions and evaluation records used by the paper.

| Directory | Contents |
| --- | --- |
| `00_PROTOCOL/` | Dataset views and comparison settings |
| `01_YOLO_SERIES/` | YOLOv8-L, YOLO11-L, YOLOv10-M and YOLOv12-M assets |
| `02_RT_DETR_R18/` | Native RT-DETR-R18 source and three-seed results |
| `03_D_FINE_M/` | D-FINE-M source and three-seed results |
| `10_YOLOV9_C/` | YOLOv9-C source and three-seed results |
| `06_COMPARATIVE_RESULTS/` | Aggregated comparison results |
| `07_PAPER_TABLES/` | Paper-ready numerical tables |

DEIM and HiLoR-DETR share the main-host assets in `../main_host/`; the RT-DETR
adaptation is in `../second_host/`. Public YOLOv8-L and YOLO11-L checkpoints are
reported as single-checkpoint comparisons, not three-seed experiments. Obtain
these two checkpoints from their provider using the release's
`EXTERNAL_ASSETS.json` and `fetch_external_assets.py` instructions.

Use `../PAPER_ASSET_MAP.md` for the table-to-file mapping and
`../ENVIRONMENT_MATRIX.md` for the experiment environments. File integrity is
covered by the optional release-wide inventory; no separate directory-layout
receipt is needed to run these experiments.
