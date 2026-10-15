# CDW1630 density/overlap stratified evaluation

This child package reports a post-hoc condition-stratified comparison of B0
and the Full model on the official CDW1630 Test split. It is intentionally
separate from the parent package's conservative dataset-description protocol.

Density and overlap thresholds were frozen using Train annotations only.
Density is the number of GT objects in an image. For each GT object, overlap is
the maximum intersection-over-own-area with another GT box; the image score is
the mean of those object scores. Low/high subsets contain whole images and all
their GT objects. No GT object is selectively removed.

The package contains B0 and Full prediction lists for detector seeds 0, 1, and
2. Each list first passed an exact full-Test metric replay against the frozen
formal evaluation. The subgroup table then pairs B0 and Full within seed and
reports three-seed means and sample standard deviations. Test data are used
only for final subgroup reporting, never for training, fitting, checkpoint
selection, threshold selection, or hyperparameter selection.

`summary_table.csv` is the paper-facing four-row table;
`per_seed_results.csv` retains every paired seed result. `results.json` and
`test_image_groups.json` preserve the complete machine-readable evidence.

CPU verification from the stored predictions:

```bash
python scripts/verify_child.py --package . --rerun
```

Claim boundary: this analysis supports condition-specific behavior of the
complete system. It does not isolate a causal contribution from UFD, LCCT,
QCCR, or UMR.
