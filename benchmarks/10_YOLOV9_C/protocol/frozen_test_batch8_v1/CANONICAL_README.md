# YOLOv9-C frozen Test protocol

This is the canonical reviewer-facing Test runner installed with the final YOLOv9-C SOTA result.

The three checkpoints were selected independently using validation AP before Test access. Test inference uses the official pinned YOLOv9 source at commit `5b10b5099005a45332ebe7669f8480d5fc791a65`, its dual-head evaluation branch `model(images)[0][1]`, official `non_max_suppression`, and official `scale_boxes`.

Accuracy protocol: RTX 4090, `640x640`, actual batch 8 (29 full batches and one batch of 5 for 237 images), FP32, confidence 0.001, NMS IoU 0.7, max detections 300. Metrics use `faster-coco-eval==1.7.2`, bbox IoU, and maxDets 1/10/100.

Reproduce one already selected seed from the project root:

```bash
bash benchmarks/10_YOLOV9_C/protocol/frozen_test_batch8_v1/reproduce_test.sh 0 /tmp/yolov9_seed0_test
```

Seeds are `0`, `1`, and `2`. The output directory must not already exist. Test results are not used for checkpoint or parameter selection.
