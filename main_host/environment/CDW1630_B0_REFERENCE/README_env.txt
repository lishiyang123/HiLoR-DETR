Environment

Conda:
deim_cvpr2025

Framework:
PyTorch + CUDA

Model:
DEIM (HGNetV2-M)

Hardware:
NVIDIA RTX 4090

Purpose:
Reproduction of CDW1630 official B0 baseline.

Canonical seed0 training precision:
FP32 (AMP disabled). The canonical checkpoint contains no scaler state.

The portable training entry point explicitly applies use_amp=False so this
protocol does not depend on argparse defaults inherited from upstream DEIM.
