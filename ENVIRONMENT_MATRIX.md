# Environment evidence matrix

The release intentionally records exact package-specific environments instead of claiming
that every external model shares one software stack.

| Asset | Primary environment evidence |
|---|---|
| CDW1630 main B0/Ours | `main_host/environment/CDW1630_B0_REFERENCE/` and per-run records |
| Cumulative ablation | `ablations/environment/main_package_snapshot/` |
| BTC generalization | `generalization/btc/environment/DEIM_RUNTIME_REFERENCE/` |
| SOTA models | model-specific `environment/` directories inside `benchmarks/` |
| Parameter sensitivity | `sensitivity/environment/` |
| Bootstrap | `bootstrap/environment/` |
| Dataset analysis | `dataset_analysis/environment/` |

## Study environments

| Study | Runtime / evidence |
|---|---|
| RT-DETR-R18 adapter | rtdetr_sota: Torch 2.0.1+cu118; Native/GCCS training Val and formal Test entries use faster-coco-eval 1.7.2 |
| Main-host efficiency | deim_cvpr2025 / Torch 2.5.1+cu124; per-measurement environment and GPU snapshots under final efficiency outputs |
| Native/adapted RT complexity | rtdetr_sota; calflops 0.3.2, 640x640; profiler bundled with its metadata |
| Closest baselines | `ranking_comparison/requirements.txt`; main-host torch environment for formal replay; CPU replay duration is not GPU latency |
| Offline budget proxy | Python, NumPy, Matplotlib; saved predictions only |

Formal Test entries use faster-coco-eval 1.7.2 under the standard COCO bbox
definitions. RT Native and GCCS use this backend for Val checkpoint selection
and Train/Val capture statistics as well. The original model-specific data
loading and COCO record preparation remain unchanged. Python/Torch environments
remain host-specific. CPU packaging is not CUDA retesting.
