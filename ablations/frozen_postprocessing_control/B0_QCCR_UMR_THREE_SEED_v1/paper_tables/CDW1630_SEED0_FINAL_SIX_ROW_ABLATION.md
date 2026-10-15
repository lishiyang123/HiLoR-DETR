# CDW1630 final six-row component ablation / seed0 Test

| Variant | UFD | LCCT | QCCR | UMR | AP | AP50 | AP75 | AR100 | Reference | Delta AP |
|---|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---|---:|
| B0 | - | - | - | - | 77.506 | 97.977 | 89.932 | 83.946 | - | +0.000 |
| B0 + QCCR | - | - | yes | - | 77.506 | 97.977 | 89.932 | 83.946 | B0 | +0.000 |
| B0 + QCCR + UMR | - | - | yes | yes | 77.552 | 97.943 | 89.955 | 83.966 | B0_QCCR | +0.045 |
| GCCS | yes | yes | - | - | 78.673 | 97.648 | 90.692 | 87.368 | B0 | +1.166 |
| GCCS + QCCR | yes | yes | yes | - | 78.782 | 97.872 | 90.844 | 87.368 | UFD_LCCT | +0.109 |
| Full model | yes | yes | yes | yes | 78.923 | 97.919 | 91.013 | 87.368 | UFD_LCCT_QCCR | +0.141 |

All rows use the fixed seed0 protocol. Delta AP uses the explicit reference column.
Test was used for final reporting only.
