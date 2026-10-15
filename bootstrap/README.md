# Bootstrap statistical analysis reproduction package v1

This package contains the reviewer-facing paired image-level Bootstrap for the
canonical CDW1630 seed0 B0/Ours comparison. It copies only the annotation and
metric-exact prediction lists needed to rerun the statistic. The original
formal evals, qualitative prediction package, and three-seed context are bound
by SHA-256.

The Bootstrap measures Test-image sampling uncertainty conditional on the
trained seed0 pair. Training-seed variability remains a distinct three-seed
mean/standard-deviation result. This distinction is deliberate and is stated
in every summary. COCO matching and accumulation use the final source's
`faster-coco-eval` backend. No GPU, inference, fitting, or parameter selection
is used.

This release retains numerical analysis only; generated plots and plotting functions are excluded.
