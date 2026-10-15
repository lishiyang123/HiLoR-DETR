# CDW1630 B0 + QCCR + UMR control

This package supplies the missing postprocessing control beside the existing
CDW1630 ablations. It compares the already released B0 detector against the
same frozen detector with QCCR and UMR enabled. Detector training is never run.

## Fixed protocol

- Detector checkpoints: the existing Val-selected B0 checkpoints for seeds
  0, 1 and 2.
- QCCR: fitted separately for each checkpoint on Val; five image-level folds;
  fold seed 0; fixed regularization `1e-3`.
- UMR: fitted separately for each checkpoint on Train; head seed 0; positive
  weight 4.
- Fusion: `tau=0`, `beta=2`, `topk=300`.
- Test: unavailable to the fit/freeze stage and used only once all three
  checkpoint-specific QCCR/UMR assets are frozen.
- Rows: `B0` and `B0 + QCCR + UMR`.

The package imports the released clean-source implementations. The two small
wrappers narrow the released fitters to the single predeclared QCCR and UMR
settings; they do not replace the fitting algorithms.

## Run

From the project root:

```bash
bash CDW_B0_QCCR_UMR_CONTROL_v1/run_server.sh --preflight
bash CDW_B0_QCCR_UMR_CONTROL_v1/run_server.sh --prepare
bash CDW_B0_QCCR_UMR_CONTROL_v1/run_server.sh --start-fit
bash CDW_B0_QCCR_UMR_CONTROL_v1/run_server.sh --fit-status
```

After `--fit-status` reports `COMPLETE`:

```bash
bash CDW_B0_QCCR_UMR_CONTROL_v1/run_server.sh --verify-frozen
bash CDW_B0_QCCR_UMR_CONTROL_v1/run_server.sh --prepare-test
bash CDW_B0_QCCR_UMR_CONTROL_v1/run_server.sh --start-test
bash CDW_B0_QCCR_UMR_CONTROL_v1/run_server.sh --test-status
```

After the Test queue reports `COMPLETE`:

```bash
bash CDW_B0_QCCR_UMR_CONTROL_v1/run_server.sh --verify
bash CDW_B0_QCCR_UMR_CONTROL_v1/run_server.sh --summary
```

Runtime products are written to
`CDW_B0_QCCR_UMR_CONTROL_STATE_v1/`, outside this source package. The final
release child package should later retain only the frozen assets, three formal
Test results, compact bindings and tables; captures, logs and temporary fit
files are runtime residue and should not be installed.
