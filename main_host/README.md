# CDW1630 main-host reproduction

The shared implementation is `source/FINAL_CODE/`. Detector seeds 0, 1 and 2
each have four configurations: Native DEIM, DEIM+CCVR, GCA and complete
HiLoR-DETR. DEIM+CCVR reuses the selected Native detector and adds its own fitted
mapping and value head. GCA and complete HiLoR-DETR share the GCA detector.

From the release root, in the main-host environment:

```bash
python -B main_host/source/FINAL_CODE/reproduce.py --release-root . \
  --profile cdw_deim_ccvr_seed0_test --output /new/external/deim_ccvr_seed0 --compare-reference
```

Use `cdw_deim_seedN_test`, `cdw_deim_ccvr_seedN_test`, `cdw_gca_seedN_test`
or `cdw_full_seedN_test` for N=0/1/2. Accuracy evaluation uses FP32 EMA,
batch 8, global Top-300 and faster-coco-eval 1.7.2.

## Assets

| Location | Contents |
|---|---|
| `checkpoints/CDW1630/B0/seedN/` | Native detector and its `calibration/` for DEIM+CCVR |
| `checkpoints/CDW1630/Ours/seedN/` | GCA detector and its `calibration/` for full HiLoR |
| `results/CDW1630/` | Twelve formal evaluations, metrics and prediction references |
| `paper_tables/three_seed/` | Main results, four-configuration results, sample SDs and paired differences |
| `training_records/CDW1630/` | Detector training records and effective configurations |
| `features/` | Optional Test and reusable Train/Val captures |
| `manifests/CDW1630_THREE_SEED_RELEASE.json` | Shared checkpoint, fitted-asset, evaluation and capture paths |

Recompute the main and system-level tables from saved evaluations on CPU:

```bash
python -B main_host/summarize_results.py --output /new/external/main_tables
```

The three-stage `stage_*.csv` files describe DEIM, GCA and full HiLoR. The
`system_*.csv` files also include DEIM+CCVR. Requires PyTorch, NumPy and
faster-coco-eval 1.7.2; no detector inference or new fitting is performed.

## Training and fitting

Use release-root `reproduce_pipeline.py` as described in the root quick start.
For DEIM+CCVR, select `--host deim --dataset cdw --method native-ccvr --seed N`.
Detector checkpoints maximize Val AP, retaining the earlier epoch in an exact
tie. Each checkpoint supplies its own Train/Val captures. Head epochs minimize
the internal Train holdout loss; official Val fits the competition mapping.
During detector training, the upstream log key `test_coco_eval_bbox` describes
the configured Val loader, not the final Test split.

## Shared analysis inputs

DEIM/HiLoR Test predictions are shared with `dataset_analysis/predictions/`.
Grouping, per-class evaluation, Bootstrap and AP-IoU analysis reuse them.
GCA seed-0 Train/Val captures are shared with `sensitivity/tau_beta/inputs/`.
The index also includes Native Train/Val captures for all three seeds.

Raw Test captures are optional feature downloads. They retain all 300 query
boxes/logits, decoder features and final detections. Captures use normalized
cxcywh boxes and `orig_size=[height,width]`; final boxes are absolute xyxy.
`analysis/ccvr/` reuses the formal seed-0 inputs for coefficient, stage-increment
and matching diagnostics. Its mapping-stage reference is under
`results/CDW1630/Ours/seed0/mapping_test/`.
