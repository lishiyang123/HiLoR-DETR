# Current status

Date: 2026-10-07

Scope: isolated nearest-neighbour comparison workspace.  The comparison run
loads the formal detector and calibration assets read-only and writes only in
this study directory.

## Bound seed-0 evidence

- Detector checkpoint SHA-256:
  `7774e65d01e0bfe1774022095dc29c6f60bd75b9f535fee3eb850de0452d2885`
- Frozen Val capture SHA-256:
  `62594e1900e255fb1ed295b9356471935a0c562de5c5c10918edd1e68f55ade6b`
- Frozen Test capture SHA-256:
  `b033ad0173c5b525a1a2edc3849f0154b55e7e85a3f0778699fc22615198f762`
- Test annotation SHA-256:
  `8130fa5706cd260a49df43d73074b2f3a55a5d2e8525c1f1b7489005a0524e05`
- Split: CDW1630 Val, 249 images, global Top-300 from all 300 x 5
  query-class candidates.
- The comparison below is a seed-0 Test result, not the paper's three-seed
  Test mean.

## Frozen seed-0 Test comparison

The Test capture contains 237 images, 1,071 ground-truth objects, and exactly
300 queries per image.  Strict checkpoint loading reported no missing or
unexpected keys.  Native/manual score and box maximum errors were both zero.
Every path used the same `300 x 5` candidate pool and deterministic global
Top-300 selection.

| Method | Setting | AP | AP50 | AP75 | AP85 | AP90 | AP95 | AR100 | Delta AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| QCCR--UMR | frozen formal assets | 78.9251 | 97.9188 | 91.0184 | 72.3820 | 47.9567 | 11.4189 | 87.3557 | +0.2530 |
| Gaussian Soft-NMS | sigma=2.0 | 78.8781 | 97.8730 | 91.1166 | 72.0783 | 47.6012 | 11.5055 | 86.9193 | +0.2060 |
| QCCR | frozen formal parameters | 78.7843 | 97.8729 | 90.8513 | 72.1203 | 47.7322 | 11.3376 | 87.3557 | +0.1122 |
| Linear Soft-NMS | IoU=0.95 | 78.7788 | 97.7032 | 90.9271 | 72.4159 | 47.7260 | 11.1298 | 86.9720 | +0.1066 |
| Native | -- | 78.6721 | 97.6432 | 90.6964 | 72.0771 | 47.7795 | 11.3793 | 87.3557 | +0.0000 |
| Gilg complete pool | -- | 78.6300 | 97.9072 | 90.9631 | 71.5807 | 46.7664 | 10.7664 | 87.2714 | -0.0421 |
| Gilg reference | near-duplicate IoU=0.99 | 78.6276 | 97.9077 | 90.9658 | 71.5577 | 46.7498 | 10.7684 | 87.2513 | -0.0445 |
| IoU-regression rescore | Train fit / Val BCE selected | 77.8469 | 96.8860 | 89.7188 | 71.3449 | 47.0836 | 11.0089 | 87.2754 | -0.8252 |

QCCR--UMR ranks first in Test AP, 0.0470 pp above Gaussian Soft-NMS and
0.2530 pp above Native.  Gaussian Soft-NMS is slightly higher at AP75 and
AP95, while QCCR--UMR is higher in overall AP, AP50, AP80, AP85, AP90, and
AR100.  The Val-frozen Soft-NMS settings must not be retuned from these Test
results.  Gilg and direct IoU multiplication are retained as negative
controls.

## Val-selected comparison

| Method | Setting | AP | AP75 | AP90 | AP95 | AR100 | Delta AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native | -- | 78.3969 | 90.0542 | 50.0749 | 11.0629 | 87.5682 | +0.0000 |
| IoU-regression rescore | Train fit / Val BCE selected | 77.5457 | 89.1541 | 49.6355 | 11.0222 | 87.4823 | -0.8513 |
| Gilg reference | near-duplicate IoU=0.99 | 78.4620 | 90.4288 | 48.6415 | 10.5027 | 87.4816 | +0.0651 |
| Gilg complete pool | -- | 78.4691 | 90.4240 | 48.6659 | 10.5079 | 87.4972 | +0.0721 |
| QCCR | frozen formal parameters | 78.5128 | 90.2172 | 50.0313 | 11.0153 | 87.5682 | +0.1159 |
| Linear Soft-NMS | IoU=0.95 | 78.5352 | 90.3081 | 50.0497 | 10.8533 | 87.2313 | +0.1383 |
| QCCR--UMR | frozen formal assets | 78.7285 | 90.4310 | 50.3135 | 11.3770 | 87.5682 | +0.3315 |
| Gaussian Soft-NMS | sigma=2.0 | 78.7690 | 90.6191 | 49.9813 | 11.0388 | 87.2249 | +0.3720 |

Gaussian Soft-NMS is 0.0405 pp above QCCR--UMR in Val AP, while QCCR--UMR
retains higher AP90, AP95, and AR100.  This is not a final method decision:
the project decision is based on the same frozen seed-0 Test stream.

The simple `score x predicted IoU` baseline is 0.8513 pp below Native on Val.
It is retained as a disclosed negative control rather than tuned away.

All 20 disclosed Val configurations remain in
`results/val/aggregate_final/all_results.json` on the experiment server.
The selected Soft-NMS settings are recorded in
`state/selection/soft_nms_seed0_val.json`.
The current lightweight fitted states, summaries, selections, and logs are
mirrored locally in
`CDW_NEAREST_NEIGHBOR_BASELINES_SEED0_VAL_RESULTS_v2.tgz` and bound by
`RESULTS_MANIFEST.sha256`; the original archive is preserved, while large
per-detection replay files stay on the server.

The completed Test capture binding, eight-path summaries, aggregate tables,
and logs are mirrored in
`CDW_NEAREST_NEIGHBOR_BASELINES_SEED0_TEST_RESULTS_v3.tgz`.  The 76 MiB
capture and large per-detection JSON files remain on the experiment server;
their hashes and provenance are included in the lightweight archive.

## Verification

- 18 CPU unit tests pass locally and on the experiment server.
- The IoU head was fit on the frozen Train capture (1,144 images), selected by
  Val BCE at epoch 20 (`0.07528674`), and never fit or selected on Test.
- The formal QCCR, UMR metadata, UMR weight, detector checkpoint, and capture
  hashes are checked before replay.
- Native replay exactly reproduces the frozen Val baseline.
- QCCR and QCCR--UMR do not alter boxes; QCCR--UMR preserves native AR100 on
  this Val capture.
- CPU replay timings are diagnostic only and are not formal deployment
  latency measurements.

## Remaining work

The seed-0 Test capture and all eight replay paths are complete.  Remaining
work is limited to formal end-to-end GPU latency measurement and, if required
for a final table, repeating the frozen comparison for the other detector
seeds.  Capture-replay timing must not be reported as deployment latency.
