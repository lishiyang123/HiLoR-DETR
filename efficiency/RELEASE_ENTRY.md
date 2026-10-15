# Final efficiency evidence

Use `publication/`, not an arbitrary dated output directory:

- `publication/COMPLEXITY_TABLE.csv`: 10 SOTA configurations, including both
  complete HiLoR hosts. Seven retained base rows plus the three accepted new
  complexity measurements. DEIM+GCCS has the same deployed detector complexity
  as DEIM and is also available in the per-mode measurement.
- `publication/LATENCY_TABLE.csv`: final balanced within-process comparison of
  DEIM / DEIM+GCCS / complete HiLoR. No SOTA-wide FPS claim.
- `publication/TRAINING_COST_TABLE.csv`: two complete 102-epoch runs, DEIM vs
  DEIM+GCCS; mean epoch time and training peak allocated VRAM.
- `publication/FINAL_RESULTS_INDEX.json`: exact authoritative measurement paths
  and their SHA bindings.

Complexity: deployed parameters, input 1x3x640x640, calflops 0.3.2, 2 FLOPs/MAC;
GFLOPs cover detector forward only and exclude postprocessing. Training-only
auxiliary geometry heads and bypassed retired deployment branches are removed.

Final latency: RTX 4090, batch 1, all 237 real Test images, 50 warmups, three
balanced alternating passes; FP16 detector / FP32 postprocessing. GPU inputs
are prepared before timing. Timed scope includes conversion to FP32, native
Top-300 or complete QCCR-UMR and coordinate scaling, but excludes IO, decode,
preprocessing, H2D and COCO evaluation. Sequential latencies retained in dated
historical outputs are superseded, not additional current results.

Training: same GPU, seed 0, batch 8, FP32, EMA; all 102 train_one_epoch calls.
Data load/H2D/forward/loss/backward/optimizer/EMA/logging are timed; validation
and checkpoint saving are excluded. Peak is maximum training allocated CUDA
memory, not reserved memory and not inference memory.

The final additional postprocessing is about 0.361 ms. Old and new timings have
different baselines: do not claim a controlled 2.7-to-0.36 ms optimization speedup.
Intermediate profiling weights are omitted from the copy; their logs and epoch
records remain, and the external originals are untouched.

Scripts and root-token configs are included. `scripts/scoped_common.py` has a
documented path-only relocation patch; no measurement/math source is changed.
calflops 0.3.2 is bundled under `vendor/` with its distribution metadata.
On another machine set `EFFICIENCY_ENV_ROOT` to a folder containing the required
`deim_cvpr2025/bin/python` and `rtdetr_sota/bin/python` environments (or install
matching environment aliases). The original separate host environments remain.

Run from this directory; every measurement writes a fresh output:

```bash
python -B reproduce_efficiency.py --stage check --workdir /your/new/efficiency_check
python -B reproduce_efficiency.py --stage smoke --workdir /your/new/efficiency_smoke
python -B reproduce_efficiency.py --stage inference --workdir /your/new/efficiency_inference
python -B reproduce_efficiency.py --stage latency --workdir /your/new/efficiency_latency
python -B reproduce_efficiency.py --stage training --workdir /your/new/efficiency_training
```

GPU stages are explicit, never automatically triggered by archive verification.
The original `SCOPE_ORIGINAL.json`/dated `STATUS.json` are execution-time records; their
old publication_ready flags are not the integrated release's archival status.
The root seven-row `RESULTS.*` files remain historical base complexity inputs.

The release wrapper uses a thin working copy outside the frozen release. Model sources and weights are read-only; cache, generated config, SCOPE/manifest updates and new outputs stay in that new copy. Current SCOPE/package manifest reflect integration, and original snapshots are retained.
