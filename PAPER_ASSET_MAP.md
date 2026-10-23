# Paper-to-asset map

This index maps manuscript claims to the authoritative release assets. Historical logs and
provenance records are evidence, not alternate manuscript result sources.

| Manuscript item | Authoritative asset | Scope |
|---|---|---|
| Dataset characteristics and split | `dataset_analysis/` and `main_host/datasets/CDW1630/` | CDW1630, 5 classes, Train/Val/Test = 1042/237/237 |
| CDW1630 B0/Ours main accuracy | `main_host/paper_tables/three_seed/` | seeds 0/1/2, mean and sample standard deviation |
| Main-host Native/GCCS/Full stages | `main_host/paper_tables/three_seed/stage_summary.csv` and `stage_deltas.csv` | nine saved formal evaluations; all 12 COCO and AP80--AP95 metrics |
| Main-host reusable captures | `main_host/manifests/CDW1630_THREE_SEED_RELEASE.json` | all nine raw Test captures; available Train/Val captures, with shared inputs referenced rather than duplicated |
| Comparison-model assets | `benchmarks/README.md` | model-specific initialization, configurations, Val selection and Test outputs |
| Recall and strict-IoU results | `main_host/paper_tables/three_seed/` and `second_host/paper_tables/` | saved COCO recall and AP80--AP95 summaries |
| System ablation | `main_host/paper_tables/three_seed/system_summary.csv` | DEIM, DEIM+CCVR, GCCS and HiLoR; seeds 0/1/2, sample SD |
| Value-residual sensitivity | `sensitivity/tau_beta/` | delta=0.02 detector seed 0; 5x5 tau/beta grid; three value-head fitting seeds; Val |
| Paired bootstrap | `bootstrap/` | seeds 0/1/2 separately; 5000 paired image resamples each; 95% percentile intervals |
| BTC generalization | `generalization/btc/paper_tables/three_seed/` | separately retrained and fitted on BTC; not zero-shot transfer |
| F6 AP--IoU curves and paired differences | `numerical_evidence/evidence/ap_iou_curve/results/curve.csv` | DEIM/HiLoR, seeds 0/1/2; ten IoU thresholds; paired mean and sample SD; CPU entry alongside the table |
| F1 / F2 / F7 photographic examples | `numerical_evidence/NUMERICAL_EVIDENCE_INDEX.json` | BTC seed-0 post-hoc examples; F2 is a detection background, not assignment evidence |

The manuscript-authoritative CDW main result files are
`main_host/paper_tables/three_seed/CDW_FINAL_THREE_SEED_MEAN_STD.csv`
and `main_host/manifests/CDW1630_THREE_SEED_RELEASE.json`.

## Additional authoritative study assets

| Manuscript evidence | Release entry | Scope |
|---|---|---|
| Second-host Native/GCCS/Full, 12 COCO and strict IoU | `second_host/README.md` | three seeds, CDW1630; Native assets shared with SOTA |
| Params/GFLOPs | `efficiency/publication/COMPLEXITY_TABLE.csv` | 10 configurations, deployed parameters and detector-forward GFLOPs; postprocessing excluded from GFLOPs |
| Paired deployment time | `efficiency/publication/LATENCY_TABLE.csv` | DEIM/GCCS/full HiLoR; FP32 forward and complete postprocessing, batch 1; three balanced passes in one session |
| Training time/peak VRAM | `efficiency/publication/TRAINING_COST_TABLE.csv` | DEIM/GCCS, two complete 102-epoch runs; training allocated memory; validation/checkpoint writes excluded |
| Candidate-ranking baselines | `ranking_comparison/results/comparison.csv` | five external methods on the seed-0 GCCS pool; Native GCCS and complete CCVR share the main-pipeline outputs |
| Oracle geometry | `numerical_evidence/evidence/oracle/` | DEIM vs GCCS, seeds 0/1/2; all 300 final-layer query boxes, no class/score/Top-K filter; CPU entry and compact box inputs |
| Density/overlap grouping and performance | `dataset_analysis/grouping/` and `dataset_analysis/conditional_performance/` | fixed density cutoff 5; Train-derived overlap cutoff; four whole-image groups, three paired seeds |
| Per-class metrics | `main_host/results/CDW1630/classwise_metrics/` | shared formal predictions; AP/AP75/AR100, three paired seeds; CPU entry in `dataset_analysis/run.py` |
| CCVR candidate-value fitting sources (S5) | `main_host/robustness/ccvr_fitting_sources/` | Train/Val 1042/237; three detector seeds; five-fold Val-OOF and size-matched Train; CPU |
| Detector-dependent CCVR gains (S10) | `main_host/analysis/ccvr/` | seed 0 DEIM/GCCS; fitted coefficients, stage increments and COCO matching/ranking from shared formal outputs; CPU |

For all current efficiency results use `efficiency/publication/FINAL_RESULTS_INDEX.json`.
The RT three-stage table shares its Native assets with the comparison-model entry;
GCCS and complete HiLoR share one detector checkpoint per seed.
