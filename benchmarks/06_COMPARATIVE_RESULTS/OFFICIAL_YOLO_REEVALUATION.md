# Official YOLO-L frozen-Test re-evaluation

Both official single checkpoints were evaluated with an actual effective batch size of 8,
640x640 FP32 inference, unchanged native postprocessing, and faster-coco-eval 1.7.2.

| Method | AP | AP50 | AP75 | AR100 | Params (M) | Batch |
|:--|--:|--:|--:|--:|--:|--:|
| YOLOv8-L | 72.423 | 96.934 | 84.525 | 77.477 | 43.610 | 8 |
| YOLO11-L | 70.700 | 95.973 | 81.045 | 76.378 | 25.283 | 8 |

Training, fitting and Test-time model/parameter selection were not performed.
