# Paper-to-asset map

This index maps manuscript claims to the authoritative release assets. Historical logs and
provenance records are evidence, not alternate manuscript result sources.

| Manuscript item | Authoritative asset | Scope |
|---|---|---|
| Dataset characteristics and split | `dataset_analysis/` and `main_host/datasets/CDW1630/` | CDW1630, 5 classes, Train/Val/Test = 1144/249/237 |
| CDW1630 B0/Ours main accuracy | `main_host/paper_tables/three_seed/` | seeds 0/1/2, mean and sample standard deviation |
| Same-dataset SOTA comparison | `benchmarks/07_PAPER_TABLES/SOTA_NINE_MODEL_PAPER_TABLES_v1/` | official single-checkpoint YOLOv8/YOLO11 and three-seed trained models |
| Recall and strict-IoU results | `benchmarks/07_PAPER_TABLES/SOTA_NINE_MODEL_PAPER_TABLES_v1/` and `second_host/RELEASE_ENTRY.md` | saved COCO recall and AP80--AP95 summaries |
| Cumulative ablation | `ablations/` | fixed seed0; Base -> GCCS -> +QCCR -> Full |
| LCCT five-point sensitivity | `gate_sensitivity/` | Val-only selection evidence for 0.005/0.01/0.02/0.04/0.08 |
| QCCR/UMR sensitivity | `sensitivity/02_INFERENCE_SIDE/` | frozen-detector Val protocol |
| Paired bootstrap | `bootstrap/` | fixed seed0, paired image resampling, 5000 replicates |
| BTC generalization | `generalization/btc/paper_tables/three_seed/` | separately retrained and fitted on BTC; not zero-shot transfer |
| AP--IoU and sensitivity figures | `numerical_evidence/evidence/` | numerical sources, not graphical exports |
| Qualitative gains and failure cases | `qualitative_evidence/` | disclosed post-hoc report analysis; no model selection |

The manuscript-authoritative CDW main result files are
`main_host/paper_tables/three_seed/CDW_FINAL_THREE_SEED_MEAN_STD.csv`
and `main_host/manifests/CDW1630_THREE_SEED_RELEASE.json`.

## Additional authoritative study assets

| Manuscript evidence | Release entry | Scope |
|---|---|---|
| Second-host Native/GCCS/Full, 12 COCO and strict IoU | `second_host/RELEASE_ENTRY.md` | three seeds, CDW1630; Native assets shared with SOTA |
| Params/GFLOPs, paired deployment time, training time/peak VRAM | `efficiency/publication/` | 10 complexity rows; three-mode latency; two complete training-cost runs |
| Closest candidate-ranking baselines | `ranking_comparison/results/test/seed0_comparison/aggregate/` | controlled seed-0 replay, not main three-seed end-to-end numbers |
| Offline proposal-budget analysis | `proposal_budget/results/` | three seeds, K=1..10, IoU .75/.80; no robot experiment |
| Oracle geometry | `numerical_evidence/evidence/ap_iou_curve/ORACLE_GEOMETRIC_COVERAGE_REPRO_v1/` | all final-layer boxes, no class/score/Top-K filter; DEIM vs GCCS |
| Density/overlap stratification | `dataset_analysis/conditional_performance/DENSITY_OVERLAP_STRATIFIED_EVAL_v1/` | four Test subsets, paired three-seed metrics |
| Per-class metrics | `main_host/results/CDW1630/classwise_metrics/` | saved COCO precision/recall arrays, no new inference |
| CCVR candidate-value fitting sources | `main_host/robustness/ccvr_fitting_sources/` | Val evaluation, not replacement of official Test metrics |

For all current efficiency results use `efficiency/publication/FINAL_RESULTS_INDEX.json`.
