# CDW1630 dataset statistics

| Split | Images | GT objects | Mean objects/image | Median | Range |
|---|---:|---:|---:|---:|---|
| Train | 1042 | 5336 | 5.120921 | 6 | 0–9 |
| Val | 237 | 958 | 4.042194 | 4 | 1–9 |
| Test | 237 | 1071 | 4.518987 | 5 | 1–10 |
| All | 1516 | 7365 | 4.858179 | 5 | 0–10 |

| Category | Train | Val | Test | All | All (%) |
|---|---:|---:|---:|---:|---:|
| Brick | 1089 | 238 | 246 | 1573 | 21.3578 |
| Concrete | 1419 | 250 | 274 | 1943 | 26.3815 |
| Plastic Bottle | 697 | 123 | 158 | 978 | 13.2790 |
| Reinforcement | 921 | 145 | 167 | 1233 | 16.7413 |
| timber | 1210 | 202 | 226 | 1638 | 22.2403 |

| Bbox size | Instances | Percent |
|---|---:|---:|
| small | 47 | 0.638153 |
| medium | 2727 | 37.026477 |
| large | 4591 | 62.335370 |

Fixed density threshold: 5; Train overlap threshold: 0.1276452544706163.

| Test group | Images | GT objects |
|---|---:|---:|
| density_low | 138 | 403 |
| density_high | 99 | 668 |
| overlap_low | 127 | 474 |
| overlap_high | 110 | 597 |

1133 of 1516 images (74.736148%) contain boxes with positive-area overlap.

Percentages use the image or GT-object denominator stated by each table. These are annotation statistics; no model performance is calculated.
