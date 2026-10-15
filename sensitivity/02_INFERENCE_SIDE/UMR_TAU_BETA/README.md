# UMR_TAU_BETA

Completed candidate-value residual sensitivity on the frozen detector-seed0
Val outputs. Positive weight=4 and Top-K=300 remain fixed. The grid is
tau={0,0.1,0.2,0.4} by beta={0,0.25,0.5,1,2}, averaged over three lightweight
head-fit seeds 0/1/2, not three detector seeds. No detector is retrained.

Unrounded results and their inputs are in `raw_results/CURRENT/`; the summary
is `raw_results/CURRENT/UMR_TAU_BETA_SUMMARY.json`. The reporting tuple remains
tau=0, beta=2. Test is excluded from fitting and parameter selection.
