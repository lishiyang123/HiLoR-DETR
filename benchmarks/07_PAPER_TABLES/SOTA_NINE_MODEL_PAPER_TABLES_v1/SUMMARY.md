# Nine-model CDW1630 accuracy statistics

All values below are percentages. `±` is the sample standard deviation across three independently
Val-selected seeds. YOLOv8-L and YOLO11-L are official single checkpoints, so no standard
deviation is reported.

Evaluator: `faster-coco-eval==1.7.2`; bbox; maxDets=1/10/100.
External-model metrics were replayed from frozen raw predictions; B0/Ours were extracted from
their authoritative saved `COCOeval_faster` tensors.

Ground-truth size bins (COCO area rules): small=1 instances/1 images, medium=393 instances/138 images, large=677 instances/229 images
Small-object results are retained but must be interpreted with the displayed support count.

## Main Accuracy

| Model | Scope | AP | AP50 | AP75 | AR100 |
|---|---|---:|---:|---:|---:|
| YOLOv8-L | single checkpoint | 72.423 | 96.934 | 84.525 | 77.477 |
| YOLO11-L | single checkpoint | 70.700 | 95.973 | 81.045 | 76.378 |
| RT-DETR-R18 | 3 seeds | 77.334 ± 0.124 | 98.267 ± 0.223 | 89.543 ± 0.651 | 84.285 ± 0.411 |
| D-FINE-M | 3 seeds | 77.101 ± 0.400 | 97.612 ± 0.199 | 88.884 ± 0.297 | 84.638 ± 0.580 |
| YOLOv10-M | 3 seeds | 74.008 ± 0.551 | 96.897 ± 0.243 | 85.296 ± 0.725 | 79.201 ± 0.317 |
| YOLOv12-M (turbo) | 3 seeds | 74.892 ± 0.092 | 97.504 ± 0.067 | 85.850 ± 0.431 | 79.031 ± 0.157 |
| YOLOv9-C | 3 seeds | 74.983 ± 0.104 | 97.647 ± 0.208 | 86.128 ± 0.575 | 79.251 ± 0.282 |
| B0 (HGNetv2-M) | 3 seeds | 77.500 ± 0.079 | 98.057 ± 0.079 | 89.902 ± 0.127 | 84.523 ± 0.512 |
| Ours (HGNetv2-M) | 3 seeds | 78.517 ± 0.356 | 97.575 ± 0.305 | 90.800 ± 0.188 | 87.416 ± 0.056 |

## Coco Scale

| Model | Scope | APs | APm | APl | ARs | ARm | ARl |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLOv8-L | single checkpoint | 50.000 | 63.116 | 77.452 | 50.000 | 70.037 | 81.313 |
| YOLO11-L | single checkpoint | 40.294 | 62.015 | 75.094 | 50.000 | 69.240 | 80.123 |
| RT-DETR-R18 | 3 seeds | 41.669 ± 14.436 | 67.892 ± 0.282 | 82.290 ± 0.125 | 53.333 ± 5.774 | 77.361 ± 0.708 | 88.159 ± 0.437 |
| D-FINE-M | 3 seeds | 60.003 ± 0.005 | 67.226 ± 0.183 | 81.939 ± 0.468 | 66.667 ± 11.547 | 77.669 ± 0.905 | 88.383 ± 0.495 |
| YOLOv10-M | 3 seeds | 48.379 ± 7.563 | 64.501 ± 0.263 | 78.861 ± 0.719 | 53.333 ± 5.774 | 70.878 ± 0.497 | 83.709 ± 0.362 |
| YOLOv12-M (turbo) | 3 seeds | 44.444 ± 14.175 | 66.813 ± 0.752 | 79.013 ± 0.295 | 56.667 ± 5.774 | 71.901 ± 0.347 | 82.879 ± 0.378 |
| YOLOv9-C | 3 seeds | 43.333 ± 16.073 | 66.161 ± 0.430 | 79.397 ± 0.241 | 53.333 ± 5.774 | 71.315 ± 0.685 | 83.519 ± 0.245 |
| B0 (HGNetv2-M) | 3 seeds | 61.672 ± 2.882 | 69.157 ± 0.096 | 81.693 ± 0.079 | 66.667 ± 5.774 | 78.064 ± 0.707 | 87.951 ± 0.441 |
| Ours (HGNetv2-M) | 3 seeds | 60.714 ± 0.714 | 70.100 ± 0.246 | 82.629 ± 0.497 | 66.667 ± 5.774 | 81.379 ± 0.194 | 90.675 ± 0.135 |

## High Recall

| Model | Scope | AR1 | AR10 | AR100 | ARs | ARm | ARl |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLOv8-L | single checkpoint | 40.105 | 77.303 | 77.477 | 50.000 | 70.037 | 81.313 |
| YOLO11-L | single checkpoint | 39.297 | 75.893 | 76.378 | 50.000 | 69.240 | 80.123 |
| RT-DETR-R18 | 3 seeds | 41.763 ± 0.133 | 81.721 ± 0.122 | 84.285 ± 0.411 | 53.333 ± 5.774 | 77.361 ± 0.708 | 88.159 ± 0.437 |
| D-FINE-M | 3 seeds | 41.737 ± 0.211 | 81.844 ± 0.394 | 84.638 ± 0.580 | 66.667 ± 11.547 | 77.669 ± 0.905 | 88.383 ± 0.495 |
| YOLOv10-M | 3 seeds | 40.860 ± 0.213 | 79.139 ± 0.353 | 79.201 ± 0.317 | 53.333 ± 5.774 | 70.878 ± 0.497 | 83.709 ± 0.362 |
| YOLOv12-M (turbo) | 3 seeds | 41.012 ± 0.054 | 78.977 ± 0.166 | 79.031 ± 0.157 | 56.667 ± 5.774 | 71.901 ± 0.347 | 82.879 ± 0.378 |
| YOLOv9-C | 3 seeds | 41.054 ± 0.072 | 79.165 ± 0.290 | 79.251 ± 0.282 | 53.333 ± 5.774 | 71.315 ± 0.685 | 83.519 ± 0.245 |
| B0 (HGNetv2-M) | 3 seeds | 42.026 ± 0.065 | 82.036 ± 0.114 | 84.523 ± 0.512 | 66.667 ± 5.774 | 78.064 ± 0.707 | 87.951 ± 0.441 |
| Ours (HGNetv2-M) | 3 seeds | 41.741 ± 0.148 | 85.068 ± 0.192 | 87.416 ± 0.056 | 66.667 ± 5.774 | 81.379 ± 0.194 | 90.675 ± 0.135 |

## High Quality

| Model | Scope | AP80 | AP85 | AP90 | AP95 |
|---|---|---:|---:|---:|---:|
| YOLOv8-L | single checkpoint | 73.808 | 58.350 | 32.468 | 4.712 |
| YOLO11-L | single checkpoint | 69.969 | 54.441 | 34.697 | 3.328 |
| RT-DETR-R18 | 3 seeds | 80.769 ± 0.347 | 67.656 ± 0.601 | 45.178 ± 0.715 | 9.058 ± 0.426 |
| D-FINE-M | 3 seeds | 80.873 ± 0.480 | 67.993 ± 0.694 | 44.692 ± 0.925 | 9.437 ± 0.921 |
| YOLOv10-M | 3 seeds | 74.957 ± 0.867 | 62.354 ± 1.854 | 38.458 ± 2.697 | 6.963 ± 0.684 |
| YOLOv12-M (turbo) | 3 seeds | 76.977 ± 0.892 | 63.525 ± 0.302 | 40.388 ± 0.589 | 6.403 ± 0.363 |
| YOLOv9-C | 3 seeds | 76.671 ± 0.462 | 63.930 ± 0.736 | 39.538 ± 0.893 | 7.087 ± 1.508 |
| B0 (HGNetv2-M) | 3 seeds | 81.432 ± 0.408 | 68.028 ± 0.592 | 43.561 ± 0.838 | 9.559 ± 0.398 |
| Ours (HGNetv2-M) | 3 seeds | 83.641 ± 0.332 | 71.955 ± 0.401 | 46.890 ± 1.029 | 10.826 ± 0.723 |
