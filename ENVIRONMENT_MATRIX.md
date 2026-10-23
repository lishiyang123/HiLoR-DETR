# Environment evidence matrix

The release intentionally records exact package-specific environments instead of claiming
that every external model shares one software stack.

| Asset | Primary environment evidence |
|---|---|
| CDW1630 main B0/Ours | `main_host/environment/CDW1630_B0_REFERENCE/` and per-run records |
| Main-host saved-evaluation tables | Main-host environment on CPU; PyTorch, NumPy and faster-coco-eval 1.7.2; `main_host/summarize_results.py` |
| Four-configuration system comparison | Main-host environment; `main_host/summarize_results.py` |
| BTC generalization | `generalization/btc/environment/DEIM_RUNTIME_REFERENCE/` |
| SOTA models | model-specific `environment/` directories inside `benchmarks/` |
| Parameter sensitivity | `sensitivity/environment/` |
| Bootstrap | CPU; Python 3.11.9, NumPy 2.4.4, faster-coco-eval 1.7.2; `bootstrap/README.md` |
| AP-IoU curves | CPU; Python 3.11.9, NumPy 2.4.4, faster-coco-eval 1.7.2; `numerical_evidence/evidence/ap_iou_curve/README.md` |
| Oracle geometry | CPU; Python 3.11.9 and NumPy 2.4.4; `numerical_evidence/evidence/oracle/README.md`; no PyTorch or COCO evaluator needed for compact-box replay |
| Dataset analysis | Python 3.8+ standard library for annotation statistics; NumPy and faster-coco-eval 1.7.2 for CPU group/per-class evaluation; `dataset_analysis/README.md` |

## Study environments

| Study | Runtime / evidence |
|---|---|
| RT-DETR-R18 adapter | [Installation](second_host/INSTALL.md), `second_host/requirements_runtime.txt`: Torch 2.0.1+cu118; Native/GCCS training Val and formal Test entries use faster-coco-eval 1.7.2 |
| Main-host paired latency | deim_cvpr2025 / Torch 2.5.1+cu124, CUDA runtime 12.4; RTX 4090, Xeon Gold 6430, driver 560.35.03; FP32 forward/postprocessing with autocast disabled; [protocol](efficiency/README.md#main-host-latency) and per-measurement environment/GPU snapshots |
| Main-host complexity and training cost | Separate measurement records under `efficiency/outputs/complexity/` and the training-cost source named in `efficiency/publication/FINAL_RESULTS_INDEX.json`; their settings are independent of the paired-latency run |
| Native/adapted RT complexity | rtdetr_sota; calflops 0.3.2, 640x640; profiler bundled with its metadata |
| Candidate-ranking baselines | `ranking_comparison/requirements.txt`; Torch 2.5.1+cu124, CUDA FP32 candidate postprocessing; faster-coco-eval 1.7.2; saved-prediction metrics can be recomputed on CPU |
| Offline budget proxy | Python, NumPy, Matplotlib; saved predictions only |

Formal Test entries use faster-coco-eval 1.7.2 under the standard COCO bbox
definitions. RT Native and GCCS use this backend for Val checkpoint selection
and Train/Val capture statistics as well. The original model-specific data
loading and COCO record preparation remain unchanged. Python/Torch environments
remain host-specific. CPU packaging is not CUDA retesting.
