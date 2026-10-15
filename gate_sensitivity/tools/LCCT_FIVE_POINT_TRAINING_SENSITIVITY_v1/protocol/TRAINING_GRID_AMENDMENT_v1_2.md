# LCCT delta five-point sensitivity amendment v1.2

Status: **FROZEN BEFORE FIVE-POINT MEASUREMENT**

The earlier sensitivity protocol reserved the three values `0.01, 0.02,
0.04`. Before training the pending points, v1.2 expands this diagnostic to the
log-spaced five-point grid `0.005, 0.01, 0.02, 0.04, 0.08`. This produces a
clearer robustness curve on both sides of the already frozen center without
changing the final model or using the Test split.

Only `lcct_delta` changes. Every run uses detector seed 0, the exact final
source and UFD asset, the same pretrained initializer, Train for fitting, Val
for all checkpoint selection and reporting, FP32, batch size 8, and 102 epochs.
The exact final `delta=0.02` center is reused only after its source, log and
checkpoint bindings pass. Four new runs are therefore required.

The queue order is `0.01, 0.04, 0.005, 0.08`: the two values already covered by
the v1.0 design are completed first, followed by the new outer points. Results
are reported as sensitivity evidence only. They cannot replace the frozen
final setting. The historical v1.0 and v1.1 decisions remain documented.

The upstream training log calls the Val COCO vector
`test_coco_eval_bbox`; this is a legacy variable name. The rendered
configuration binds that evaluator to `instances_val2017.json`, and the Test
split is neither materialized nor accessed by this workflow.

The final-source checkpoint schedule is unchanged during execution. After a
new value has completed all 102 epochs and its Val-selected checkpoint has
passed verification, only the reproducible `checkpointNNNN.pth` periodic
snapshots are removed. Each run retains `last.pth`, `best_stg1.pth`,
`best_stg2.pth` when the stage-2 trajectory produces one, the Val-selected
checkpoint, the complete log, Val evaluation records, and signed run and
selection receipts. This storage policy has no numerical effect.
