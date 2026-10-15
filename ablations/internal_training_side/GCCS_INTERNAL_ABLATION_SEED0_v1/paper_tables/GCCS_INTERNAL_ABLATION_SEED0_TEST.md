# GCCS training-side internal ablation / seed0 Test

| Variant | AP | AP50 | AP75 | AR100 | Delta AP vs. B0 (pp) |
|---|---:|---:|---:|---:|---:|
| b0 | 77.506 | 97.977 | 89.932 | 83.946 | +0.000 |
| ufd_without_difficulty | 77.517 | 98.015 | 88.773 | 84.671 | +0.011 |
| ufd_only | 77.649 | 98.078 | 88.822 | 84.347 | +0.143 |
| lcct_only | 78.149 | 97.627 | 90.537 | 87.172 | +0.643 |
| gccs | 78.673 | 97.648 | 90.692 | 87.368 | +1.166 |

All checkpoints were selected on Val before Test evaluation.
QCCR and UMR are disabled for every row.
