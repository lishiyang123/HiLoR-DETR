# Offline sorting-proposal budget study

The implementation and completed staged results are now one release asset.
`results/` contains 120 per-seed rows (two methods x three seeds x ten budgets
x two IoU thresholds), per-image counts, three-seed means/sample SD, and paired
Full-minus-B0 increments. Numerical curve data remain in the result tables; generated curve graphics are excluded.

Protocol: `protocol/ACTION_BUDGET_PROTOCOL_LOCK.json`. Final scored Top-300,
descending score, stable original-order ties, no score filter; K=1..10,
IoU >=0.75 and >=0.80. The main K=5 was defined from Train median density.
Same-class non-crowd GT is matched once; other queue entries remain valid,
duplicate or other-invalid under the unchanged evaluator. Exported out-of-image
boxes and finite degenerate boxes are retained, not silently clipped/deleted.

Input SHA bindings: `manifests/INPUT_BINDINGS.json`. Six official Test prediction
files are shared, without duplication, with the integrated density/overlap
subpackage at `../dataset_analysis/conditional_performance/DENSITY_OVERLAP_STRATIFIED_EVAL_v1/predictions/`.

From the release root:

```bash
python -B reviewer_studies.py check-inputs
python -B reviewer_studies.py budget-replay --output /your/new/replay/action_budget
```

This is CPU-only offline reporting, not training or detector inference. The
wrapper calls the original evaluator and requires exact numerical agreement
with archived rows. Results, including unfavorable duplicate-action changes,
remain unchanged. No robot, grasp success or closed-loop performance is measured.
Original manager/run_server staging tools are retained as historical workflow;
use the release wrapper for relocated replay. The staged-only flags in original
STATUS/results describe their initial generation, not their current installation.

`PAYLOAD_MANIFEST.sha256` reflects the release README; its original copy is retained. Evaluation source and results are unchanged.
