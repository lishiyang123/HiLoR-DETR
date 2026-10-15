# B0 frozen-postprocessing controls

This child package contains the three-seed B0 branch: B0, B0+QCCR, and
B0+QCCR+UMR. QCCR and UMR assets are fitted separately for each frozen B0
checkpoint under the predefined Train/Val protocol. The QCCR-only row reuses
the already frozen QCCR artifacts and explicitly disables UMR.

The paper tables include both the three-seed branch and a six-row seed0 table
that joins this branch to the released cumulative GCCS ablation. Test is used
only for final reporting. Runtime captures, temporary fits, queue state,
console logs and failed attempts are excluded.
