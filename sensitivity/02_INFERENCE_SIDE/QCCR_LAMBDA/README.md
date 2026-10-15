# CCVR competition-mapping sensitivity

Status: complete. The image-level five-fold Val-OOF results for primary-host
detector seed 0 are in `raw_results/CURRENT/` and `tables/QCCR_LAMBDA_OOF_VAL.csv`.
They correspond to Supplementary Fig. S1(b).

| Mapping regularization | OOF Val AP | AP increment (pp) |
|---:|---:|---:|
| 0.0001 | 78.5689 | +0.1720 |
| 0.001 | 78.4950 | +0.0981 |
| 0.01 | 78.4128 | +0.0159 |
| 0.1 | 78.3964 | -0.0005 |

The retained coefficient is 0.001, fixed in advance. None of the four candidates
satisfied every diagnostic guard. The diagnostic did not reselect the parameter
or replace formal Test results. The curve is not the three-seed value-fitting
source comparison in Supplementary Table S5.

Fitting code: `main_host/source/FINAL_CODE/tools/qccr/fit_qccr.py`.
OOF evaluation code:
`main_host/source/FINAL_CODE/tools/qccr/evaluate_qccr_oof_val.py`.
Use `reproduce_pipeline.py --stage capture` for Train/Val capture; the fixed
mapping fit records and fold assignments are released with each checkpoint.
