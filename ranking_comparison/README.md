# Seed-0 candidate-ranking comparison release

Eight completed paths share one frozen seed-0 GCCS Test capture (all 300 query
boxes x 5 class scores), deterministic global Top-300, and the same COCO replay:
Native, QCCR, QCCR-UMR, Linear/Gaussian Soft-NMS, Gilg reference/complete-pool,
and score x predicted-IoU. Boxes are unchanged by rescoring.

Authoritative table: `results/test/seed0_comparison/aggregate/all_results.json`
and its CSV/Markdown companions. All 20 attempted Val configurations remain
in `results/val/aggregate_final/`; Soft-NMS settings were frozen from Val.
Gilg is independently implemented from the reference formulation; reference
near-duplicate IoU=0.99 and complete-pool variants are both disclosed.
IoU regression was fitted from frozen Train features and selected by Val BCE.

`state/captures/` includes the original Train, Val and Test captures; fitted
Gilg and IoU-regression states are included. QCCR/UMR/formal detector assets
are shared with `../main_host/`, not refitted during integration.

This controlled-replay Native/Full table is distinct from the primary end-to-end
paper results. Its deltas use its own Native replay; never replace only Native
or Full rows with old primary metrics. CPU replay durations are diagnostics,
not deployment latency. Other detector seeds and comparative GPU deployment
timings are not completed evidence in this package.

Original protocol/history files are unchanged, including historical project
policy wording. Packaging does not assert that Test outputs were never viewed,
nor that every project decision was predeclared. Negative controls are retained.

From the release root in the main-host environment:

```bash
PYTHONPATH="$PWD/ranking_comparison/source" python -B -m unittest discover -s ranking_comparison/tests
python -B reviewer_studies.py nearest-replay --output /your/new/replay/nearest_seed0
```

The second command uses saved candidates, CPU and original evaluator scripts;
it does not run the detector, use a GPU, or choose new settings. Fitting scripts
remain available under `source/` for an optional fresh fit.

`SOURCE_MANIFEST.sha256` is refreshed for release documentation; the original source manifest is preserved separately. Algorithm source remains byte-identical.
