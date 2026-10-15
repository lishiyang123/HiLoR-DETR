# LCCT five-point training-sensitivity executor v1

This installer creates the independent result package
`gate_sensitivity`. The authoritative inputs are read
only from `main_host`; old experiment directories are not inputs.

The formal grid is `0.005, 0.01, 0.02, 0.04, 0.08`. The exact final seed-0
center (`0.02`) is reused after SHA verification, while four new seed-0 runs
are executed. Train is used for fitting and Val for checkpoint selection. Test
is prohibited.

Run each command separately and wait for it to finish:

```bash
bash LCCT_FIVE_POINT_TRAINING_SENSITIVITY_v1/run_server.sh --preflight
bash LCCT_FIVE_POINT_TRAINING_SENSITIVITY_v1/run_server.sh --prepare
bash LCCT_FIVE_POINT_TRAINING_SENSITIVITY_v1/run_server.sh --audit
bash LCCT_FIVE_POINT_TRAINING_SENSITIVITY_v1/run_server.sh --smoke
bash LCCT_FIVE_POINT_TRAINING_SENSITIVITY_v1/run_server.sh --start
```

`--start` launches a detached serial worker, so closing the terminal does not
terminate training. Check it with `--status`. If a run fails or the host is
restarted, call `--resume`; completed values are retained, while only the
incomplete value is archived and restarted from the frozen pretrained
initializer. Use `--run` instead of `--start` only when foreground execution is
desired.

The source checkpoint schedule is not changed. After each successful 102-epoch
run is verified, the executor removes only its reproducible
`checkpointNNNN.pth` periodic snapshots. It retains `last.pth`, the stage-best
checkpoints that were actually produced, the Val-selected checkpoint, the full
log, Val evaluations, and run/selection receipts. This keeps the formal package
compact without changing training or its evidence.

After all four runs finish:

```bash
bash LCCT_FIVE_POINT_TRAINING_SENSITIVITY_v1/run_server.sh --summary
bash LCCT_FIVE_POINT_TRAINING_SENSITIVITY_v1/run_server.sh --verify
```

The one-epoch smoke test is explicitly diagnostic and uses tiny Train/Val
annotation subsets. It exercises the current LCCT branch on the GPU but is
excluded from formal results. The formal configuration remains FP32, seed 0,
batch size 8 and 102 epochs.

Dependency validation follows the final source requirements exactly, including
`faster-coco-eval>=1.6.5`; no separate `pycocotools` installation is required.
