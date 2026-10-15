
# Protocol registry

This registry describes completed Val sensitivity evidence:

- Training-side: detector seed 0, delta={0.005,0.01,0.02,0.04,0.08}; the
  authoritative five-point package is `../../gate_sensitivity/`.
- Mapping regularization: the four-candidate OOF-Val diagnostic; lambda=0.001
  is the fixed protocol value, not a candidate accepted by all diagnostic guards.
- Candidate-value residual: fixed pw=4 and Top-K=300, tau-by-beta Val grid
  averaged across three head-fit seeds, without retraining the detector.

Test is not used for selecting any parameter. Results, input identities and
interpretation limits are retained in the corresponding study directories.
