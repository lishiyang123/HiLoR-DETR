# LCCT delta five-point Val sensitivity

| delta | Val AP | AP50 | AP75 | AR100 | best epoch | source |
|---:|---:|---:|---:|---:|---:|:---|
| 0.005 | 0.777499 | 0.967340 | 0.890124 | 0.876294 | 94 | new formal run |
| 0.01 | 0.779464 | 0.968378 | 0.894298 | 0.876113 | 98 | new formal run |
| 0.02 | 0.783957 | 0.972253 | 0.900541 | 0.875682 | 99 | exact final run reused |
| 0.04 | 0.781171 | 0.973477 | 0.905092 | 0.870575 | 101 | new formal run |
| 0.08 | 0.780262 | 0.976397 | 0.903378 | 0.865115 | 92 | new formal run |

Train was used for fitting and Val for checkpoint selection. Test was not accessed.
