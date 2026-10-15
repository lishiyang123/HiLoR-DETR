# CDW1630 seed0 cumulative ablation

All accuracy values are percentages. Delta AP is the incremental change from the preceding row.

| Variant | UFD-LCCT | QCCR | UMR | AP | AP50 | AP75 | AR100 | AP85 | AP90 | Delta AP (pp) |
|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | – | – | – | 77.506 | 97.977 | 89.932 | 83.946 | 67.834 | 44.113 | +0.000 |
| UFD-LCCT | ✓ | – | – | 78.673 | 97.648 | 90.692 | 87.368 | 72.070 | 47.795 | +1.166 |
| UFD-LCCT + QCCR | ✓ | ✓ | – | 78.782 | 97.872 | 90.844 | 87.368 | 72.120 | 47.735 | +0.109 |
| UFD-LCCT + QCCR + UMR | ✓ | ✓ | ✓ | 78.923 | 97.919 | 91.013 | 87.368 | 72.382 | 47.965 | +0.141 |

UFD and LCCT are independently configurable and jointly enabled in this cumulative training-stage row. Because no UFD-only or LCCT-only row is included, the table does not attribute an independent effect or strict interaction to either mechanism.
Detector checkpoints and inference parameters were selected using Train/Val only. Test was used once for final reporting and never for module or hyperparameter selection.
