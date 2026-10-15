# CDW1630 paired image-level Bootstrap protocol

Status: **frozen before Bootstrap computation**.

The canonical seed0 B0 and Ours predictions are compared with an ordinary
paired nonparametric Bootstrap over the 237 official-Test images. Every
replicate samples 237 images with replacement, and the same image
multiplicities are applied to both methods. Per-image COCO matching is cached,
but COCO accumulation and global detection-score ranking are recomputed for
every resample. This is not an average of ad-hoc per-image AP values.

- Replicates: `5000`.
- RNG: NumPy PCG64, seed `20260915`.
- Primary endpoint: paired delta COCO AP@[0.50:0.95].
- Descriptive secondary endpoints: AP50, AP75, and AR100.
- Interval: two-sided 95% percentile interval.
- Tail summary: empirical two-sided sign-tail probability with plus-one
  correction, interpreted confirmatorily only for the predeclared AP endpoint.

The image Bootstrap conditions on the canonical trained seed0 pair. Exact
seed0/1/2 results are retained separately to describe training-seed
variability; three seeds are not misrepresented as a large-sample Bootstrap.
The Test split is used only for final statistical reporting and never for
model, parameter, checkpoint, or qualitative-case selection.
